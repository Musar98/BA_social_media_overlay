# Shader-Based Image Transformations

This document describes the WebGL2 shader implementation used by the browser
overlay renderer. It also explains how the shader code translates the
reference transformations from
`python/src/social_media_overlay/ext/gebhardt/image_transformations/`.
The comparisons are checked against Kornia `0.8.2`, the version
locked in `python/uv.lock`.

## Runtime Architecture

The production shader path is implemented in
`ai-overlay-engine/src/transformation/`:

- `transformations.ts` owns the WebGL2 renderer. It creates the WebGL context,
  compiles and links shaders, uploads video frames into a texture, sends uniform
  values to the GPU, draws a full-canvas quad, and manages smoothing and
  optional metrics.
- `ClipToUv.ts` contains the vertex shader. It receives clip-space quad
  positions and converts them into normalized canvas UV coordinates.
- `ImageAdjustmentsPipeline.ts` contains the active fragment shader. It maps
  canvas UV coordinates to source-image UV coordinates, samples the video
  texture, applies the image transformations, and writes the final pixel color.
- `PostProcessor.ts` maps the flat model output into the parameter layout
  expected by the renderer and shader.
- `IdentityParams.ts` defines the identity/default parameter values used before
  the first model prediction is available.

The render loop in `renderer/renderLoop.ts` draws each valid video frame when
the filter is enabled. AI inference is performed less frequently, by default
every n valid rendered frames after the first immediate prediction. Between
predictions, the most recent parameters are reused and smoothed before they are
sent to the shader.

## Frame Rendering Flow

Each frame follows this sequence:

1. The render loop checks that the video is playable and that the filter is
   enabled.
2. `ImageTransformRenderer.renderFrame` verifies that the source dimensions are
   valid.
3. The current parameter object is normalized. Missing scalar values fall back
   to identity values, tone curve values fall back to `1.0`, and color curve
   values fall back to `1.0`.
4. If smoothing is enabled, scalar parameters and curve arrays are updated with
   a simple spring-damper step before upload.
5. The canvas size is matched to the source/display dimensions.
6. The current video frame is uploaded into `u_image`, a `TEXTURE_2D` with
   `CLAMP_TO_EDGE` wrapping and `LINEAR` minification/magnification filtering.
7. The renderer uploads uniforms such as source size, texel size, fit mode, and
   transformation parameters.
8. WebGL draws a full-canvas quad made from two triangles.
9. The vertex shader emits canvas UVs, and the fragment shader computes one
   output color per fragment.

The renderer draws into the default framebuffer of the target canvas. No
intermediate framebuffer or multi-pass production shader is currently used.

## Shader Inputs

The active fragment shader declares the following uniforms:

| Uniform | Type | Meaning |
| --- | --- | --- |
| `u_image` | `sampler2D` | Source video/image texture. |
| `u_texelSize` | `vec2` | One source texel in UV units, `(1 / width, 1 / height)`. |
| `u_canvasSize` | `vec2` | Output canvas size in pixels. |
| `u_imageSize` | `vec2` | Source image/video size in pixels. |
| `u_fitMode` | `int` | `0` for cover, `1` for contain. |
| `u_sharp` | `float` | Sharpen strength. May be forced to `0` by `SHARPEN_ENABLED`. |
| `u_exposure` | `float` | Exposure in stops. |
| `u_contrast` | `float` | Contrast multiplier. |
| `u_saturation` | `float` | Saturation multiplier. |
| `u_blur` | `float` | Kept for API/model compatibility; intentionally unused. |
| `u_imageMean` | `float` | CPU-supplied global image mean for contrast. |
| `u_toneCurve[8]` | `float[8]` | Eight tone-curve segment weights. |
| `u_colorCurve[8]` | `vec3[8]` | Eight RGB color-curve segment weights. |

The model output is a flat array of at least 37 values:

```text
0      exposure
1      saturation
2..9   tone curve, 8 values
10..33 color curve, 24 values
34     contrast
35     sharp
36     blur
```

The Python/model color curve layout is flattened as `R0..R7, G0..G7, B0..B7`.
WebGL `uniform3fv` for `vec3[8]` expects `R0,G0,B0, R1,G1,B1, ... R7,G7,B7`,
so `PostProcessor.ts` explicitly reorders the 24 color values before rendering.

The identity/default parameters are:

```text
sharp      = 1.0
exposure   = 0.0
contrast   = 1.0
saturation = 1.0
blur       = 0.0
imageMean  = 0.5
toneCurve  = [1, 1, 1, 1, 1, 1, 1, 1]
colorCurve = eight RGB triples of [1, 1, 1]
```

`imageMean` is not part of the ONNX output. It is computed in
`worker/runPredictionTask.ts` from the image data used for inference. The worker
uses the same grayscale weights as Kornia's `rgb_to_grayscale` for floating
point images:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
imageMean = mean(gray)
```

## UV Mapping and Fitting

The vertex shader receives six clip-space vertices covering the full canvas:

```text
(-1,-1), (1,-1), (-1,1), (-1,1), (1,-1), (1,1)
```

It maps clip space to canvas UV space:

```text
u = 0.5 * (x + 1.0)
v = 1.0 - 0.5 * (y + 1.0)
```

The Y coordinate is flipped so the sampled texture appears upright in the
canvas coordinate system.

The fragment shader then maps canvas UVs to source-image UVs:

- In `cover` mode, the source fills the whole canvas and may be cropped along
  one axis.
- In `contain` mode, the whole source is visible and empty letterbox/pillarbox
  regions are rendered as opaque black.

This mapping is separate from the color adjustments. It decides which source
texel is sampled for a canvas pixel before any transformation is applied.

## Reference Translation

The Python reference applies a sequence of named transformations in
`image_transformations.py`. The current browser shader implements the subset
needed by the overlay model output: exposure, saturation, tone curve, color
curve, contrast, and sharpening.

| Python reference | Kornia / PyTorch operation | GLSL implementation | Translation |
| --- | --- | --- | --- |
| `ittf.apply_exposure` | `im * 2^exposure`, then clamp | `applyExposure` | Faithful |
| `kornia.enhance.adjust_saturation` | RGB to HSV, clamp `S * factor`, HSV to RGB | `applySaturation` | Formula-equivalent up to hue units, epsilon, and floating-point details. |
| `ittf.apply_curve_adjustment` for tone | 8-step clamped segment accumulation | `applyToneCurve` | Faithful |
| `ittf.apply_curve_adjustment` for color | 8-step clamped segment accumulation per channel | `applyColorCurve` | Faithful |
| `kornia.enhance.adjust_contrast_with_mean_subtraction` | `image * factor + grayMean * (1 - factor)`, then clamp | `applyContrast` | Same affine formula; mean is supplied as a uniform. |
| `kornia.enhance.sharpness` | 3-by-3 `[1 1 1; 1 5 1; 1 1 1] / 13` depthwise convolution, border restore, blend | `applySharpen` | Same kernel family, implemented through texture sampling. |
| `kornia.filters.gaussian_blur2d` | Separable Gaussian blur, reflect padding, project wrapper uses `(25, 25)` | Not applied | Not used; Model evaluation showed neglible variance |

The exact Kornia source files used for this comparison are:

- [`kornia/enhance/adjust.py`](https://github.com/kornia/kornia/blob/v0.8.2/kornia/enhance/adjust.py)
- [`kornia/filters/gaussian.py`](https://github.com/kornia/kornia/blob/v0.8.2/kornia/filters/gaussian.py)
- [`kornia/color/gray.py`](https://github.com/kornia/kornia/blob/v0.8.2/kornia/color/gray.py)
- [`kornia/color/hsv.py`](https://github.com/kornia/kornia/blob/v0.8.2/kornia/color/hsv.py)

### Exposure

Python reference:

```python
exposed_im = im * torch.exp(exposure_param * torch.log(torch.tensor(2.0)))
exposed_im = torch.clamp(exposed_im, min=0.0, max=1.0)
```

GLSL:

```glsl
vec3 applyExposure(vec3 c, float exposureValue) {
    return clamp(c * exp2(exposureValue), 0.0f, 1.0f);
}
```
Both implementations multiply RGB by the exposure factor and clamp the result to `[0, 1]`.

### Saturation

The project wrapper calls:

```python
kornia.enhance.adjust_saturation(im, torch.clamp(saturation_param, min=0))
```

In Kornia `0.8.2`, `adjust_saturation` converts RGB to HSV, calls
`adjust_saturation_raw`, and converts back to RGB. The raw operation unpacks
`h, s, v`, computes:

```text
s_out = clamp(s * factor, 0, 1)
```

and returns `h, s_out, v`.

The shader implements the same operation explicitly:

```glsl
vec3 applySaturation(vec3 c, float satValue) {
    vec3 hsv = rgb2hsv(c);
    hsv.y = clamp(hsv.y * satValue, 0.0f, 1.0f);
    return clamp(hsv2rgb(hsv), 0.0f, 1.0f);
}
```

It is the same HSV-channel formula. The remaining non-identical parts are representational
and numerical. Kornia represents hue in `0..2pi`; the shader represents hue in
`0..1`. Kornia's RGB/HSV conversion uses PyTorch tensor operations with an
epsilon of `1e-8`; the shader uses GLSL scalar/vector operations with `1e-10`.
These differences can change tiny floating-point details, but not the intended
parameter semantics.

### Tone Curve

Python reference:

```python
curve_steps = param.shape[2]
total_image = im.new_zeros(im.size())
for i in range(curve_steps):
    offset = float(i) / float(curve_steps)
    clamped = torch.clamp(im - offset, min=0.0, max=1.0 / curve_steps)
    total_image = total_image + clamped * param_list[i]
total_image = torch.clamp(total_image, min=0.0, max=1.0)
```

GLSL:

```glsl
vec3 applyToneCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);

    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);
        total += seg * u_toneCurve[i];
    }

    return clamp(total, 0.0f, 1.0f);
}
```

This is a faithful translation for `curve_steps = 8`. Each channel value is
split into eight equal input intervals. For each interval, the implementation
computes the clamped contribution:

```text
segment_i(c) = clamp(c - i / 8, 0, 1 / 8)
```

The final tone-adjusted color is:

```text
tone(c) = clamp(sum_i segment_i(c) * toneCurve[i], 0, 1)
```

The tone curve uses one scalar weight per segment. In GLSL that scalar
broadcasts over `vec3`, matching the Python tensor shape `[B, 1, 8, 1]`.

### Color Curve

The color curve uses the same piecewise accumulation as the tone curve, but
with independent weights for red, green, and blue:

```glsl
vec3 applyColorCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);

    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);
        total += seg * u_colorCurve[i];
    }

    return clamp(total, 0.0f, 1.0f);
}
```

The Python color parameter has shape `[B, 3, 8, 1]`. The shader receives it as
`vec3 u_colorCurve[8]`, where each entry is the RGB weight triple for one
segment. After the reorder in `PostProcessor.ts`, the formula is equivalent to
the Python reference:

```text
color(c).rgb = clamp(sum_i segment_i(c.rgb) * colorCurve[i].rgb, 0, 1)
```

### Contrast

The reference code path uses Kornia mean-subtraction contrast:

```python
return kornia.enhance.adjust_contrast_with_mean_subtraction(im, contrast_param)
```

In Kornia `0.8.2`, RGB images first pass through `rgb_to_grayscale`, which uses
the floating-point weights:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

Kornia then computes the mean over height and width and applies:

```text
out = clamp(image * factor + grayMean * (1 - factor), 0, 1)
```

The shader uses the same affine form explicitly:

```glsl
vec3 applyContrast(vec3 c, float contrastValue, float imgMean) {
    return clamp(
        c * contrastValue + vec3(imgMean) * (1.0f - contrastValue),
        0.0f,
        1.0f
    );
}
```

This can also be written as:

```text
contrast(c) = clamp(mean + contrastValue * (c - mean), 0, 1)
```

which expands to the GLSL expression:

```text
c * contrastValue + mean * (1 - contrastValue)
```

The decision is that the shader does not compute the global mean itself.
Instead, `u_imageMean` is supplied by the CPU/worker side. The worker computes
that mean using Kornia's grayscale weights. This avoids a costly full-frame
reduction in the fragment shader and keeps the draw pass local to each fragment.
The contrast formula is the same as Kornia's formula;

### Sharpening

The Python reference exposes sharpening through Kornia:

```python
return kornia.enhance.sharpness(im, torch.clamp(sharp_param, min=0))
```

In Kornia `0.8.2`, `sharpness` builds this depthwise convolution kernel for
each channel:

```text
(1 / 13) *
[1 1 1]
[1 5 1]
[1 1 1]
```

It convolves the image without padding, clamps the result, pads it back to the
original size, restores the original border pixels, and blends between the
degenerate filtered image and the original input with the sharpness factor.

The production shader uses the same 3-by-3 weighted neighborhood directly at
the current UV:

```glsl
vec3 sum =
    n00 + n10 + n20 +
    n01 + 5.0f * c + n21 +
    n02 + n12 + n22;

vec3 degenerate = clamp(sum * (1.0f / 13.0f), 0.0f, 1.0f);
return clamp(mix(degenerate, c, sharpValue), 0.0f, 1.0f);
```

Border fragments are returned unchanged, and sharpening is skipped entirely when
`sharpValue <= 0.001`.

It uses the same kernel weights, the same division by `13`, the same border idea,
and the same blend direction: filtered image toward original image as the factor
increases. It is still not guaranteed to be bit-identical. Kornia performs on the tensor grid and then pads/restores borders. The
shader samples the source texture at neighboring UVs, using the renderer's
texture filtering and GPU precision rules. The project also monkey-patches
Kornia's internal `_blend_one` during export so blending is expressed as a
tensor formula; the shader naturally uses GLSL `mix` for that same linear blend.

The feature flag `SHARPEN_ENABLED` can force `u_sharp` to `0`, disabling the
effect without changing the model parameter contract.

### Blur
The Python reference applies blur at the end of the transformation sequence:

```python
sigma = torch.clamp(blur_param, min=0)
sigma = sigma.repeat(2, 1).view(-1, 2)
im = kornia.filters.gaussian_blur2d(im, kernel_size=(25, 25), sigma=sigma)
```
Kornia builds a Gaussian convolution kernel from sigma. For a pixel at the
center, neighboring pixels are weighted by their distance from that center:

```python
G_sigma(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
```

After normalization, all weights sum to `1`. The output pixel is therefore a
weighted average of nearby pixels. In the reference implementation, `blur_param` directly controls `sigma`. This
means the blur parameter changes the kernel itself:
```
small sigma  -> center-heavy kernel -> weak blur
large sigma  -> wider/flatter kernel -> stronger blur
```

With `kernel_size=(25, 25)`, Kornia evaluates this Gaussian on the discrete
coordinate grid:

```
x, y in {-12, -11, ..., 0, ..., 11, 12}
```

So Kornia has a large support region and can assign non-zero weights to pixels
up to 12 pixels away from the center in each direction. The value of `sigma`
decides how much those farther pixels matter.

An approximation could be implemented like this:
```glsl
for(int y = -2; y <= 2; y++) {
    for(int x = -2; x <= 2; x++) {
        vec2 off = vec2(float(x), float(y)) * texel * radius;
        float d2 = float(x * x + y * y);
        float w = exp(-d2 / 4.0f);

        sum += processSample(uv + off) * w;
        wsum += w;
    }
}
```
This also computes a weighted average, but the kernel is fixed. A fixed kernel
means the relative weights are always the same. The shader always uses this
5x5 coordinate grid:
```
x, y in {-2, -1, 0, 1, 2}
```

Resulting in:
```
Kornia:
  fixed coordinate grid: -12..12
  25x25 kernel

Old shader:
  fixed coordinate grid: -2..2
  5x5 kernel
```

There is also an ordering problem. In the
reference pipeline, blur is applied at the end of the transformation sequence.
That means the blur should operate on pixels that have already been transformed
by exposure, saturation, tone curve, color curve, contrast, and sharpening.
A fragment shader normally samples from the input texture. In the current
renderer, that input texture is the original video frame, not an already edited
image. Therefore, a blur pass cannot simply sample neighboring "already modified"
pixels, because those modified pixels do not exist in a texture yet.

A faithful implementation would therefore require a multi-pass rendering
pipeline:
```
Pass 1:
  render the transformed image into an intermediate texture

Pass 2:
  blur that transformed texture, with a horizontal blur pass and a
  vertical blur pass

Pass 3:
  output the blurred result to the canvas
```

This was currently not activated. The main reasons are:
```
1. Computational cost
   A blur pass requires many additional texture samples per output pixel. A
   faithful separable blur would also require intermediate textures and extra
   draw calls.

2. Architectural complexity
   The current renderer uses a single production fragment shader. Supporting
   blur after all other edits would require a framebuffer/intermediate-texture
   pipeline.

3. Limited model impact
   Model evaluation showed little variation in the predicted blur parameter.
   Since blur changed only weakly, the visual benefit did not justify the added
   rendering cost and complexity.
```

## Production Shader Order

The active shader order is:

1. Map canvas UV to source-image UV.
2. Sample the source texture.
3. Apply sharpening.
4. Apply exposure.
5. Apply saturation.
6. Apply the tone curve.
7. Apply the color curve.
8. Apply contrast around `u_imageMean`.
9. Clamp and output `vec4(color, 1.0)`.

In code, the color adjustment stack is:

```glsl
vec3 adjustColor(vec3 color) {
    color = applyExposure(color, u_exposure);
    color = applySaturation(color, u_saturation);
    color = applyToneCurve(color);
    color = applyColorCurve(color);
    color = applyContrast(color, u_contrast, u_imageMean);
    return color;
}
```

and the full effect entry point is:

```glsl
vec3 applyOptimizedEffects(vec2 uv, vec2 texel) {
    vec3 color = sampleImage(uv);
    color = applySharpen(color, uv, texel, u_sharp);
    return adjustColor(color);
}
```

## Temporal Smoothing

The renderer smooths parameters before upload by default. Each scalar parameter
and each curve element has a current value and a velocity. On each rendered
frame:

```text
velocity += (target - current) * stiffness
velocity *= damping
current += velocity
```

The default values are:

```text
stiffness = 0.05
damping   = 0.5
```

This is a low-pass temporal filter. It reduces visible jumps because AI
prediction happens only at discrete intervals while rendering continues every
video frame.