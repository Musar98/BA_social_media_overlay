#version 300 es
precision highp float;

// Interpolated UV coordinates from the vertex shader.
// These are canvas-space UVs in the range [0, 1].
in vec2 v_uv;

// Final output color for this fragment.
out vec4 outColor;

// Source image texture.
uniform sampler2D u_image;

// Size of one texel in UV space, typically:
// vec2(1.0 / imageWidth, 1.0 / imageHeight)
uniform vec2 u_texelSize;

// Size of the output canvas in pixels.
uniform vec2 u_canvasSize;

// Size of the source image in pixels.
uniform vec2 u_imageSize;

// User-controlled adjustment parameters.
//uniform float u_gamma;       // Gamma adjustment
uniform float u_sharp;       // Edge-based sharpening strength
//uniform float u_wb;          // White balance blend factor
uniform float u_exposure;    // Exposure in stops
//uniform float u_bright;      // Linear brightness offset
uniform float u_contrast;    // Contrast multiplier around 0.5
uniform float u_saturation;  // Saturation multiplier
//uniform float u_bw;          // Black-and-white mix amount
//uniform float u_hue;         // Hue rotation in radians
uniform float u_blur;        // Blur radius multiplier

// Piecewise tone curve weights.
// Each segment influences a fixed tonal range of the input color.
uniform float u_toneCurve[8];

// Piecewise per-channel color curve weights.
// Each segment is a vec3 so RGB channels can be shaped independently.
uniform vec3 u_colorCurve[8];

// Computes perceptual luminance using weighted RGB components.
// This is used when blending toward grayscale.
float rgb2lum(vec3 c) {
    return 0.27f * c.r + 0.67f * c.g + 0.06f * c.b;
}

// Converts RGB to HSV.
// Hue is stored in [0, 1), saturation and value in [0, 1].
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10f;

    return vec3(abs(q.z + (q.w - q.y) / (6.0f * d + e)), d / (q.x + e), q.x);
}

// Converts HSV back to RGB.
vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f);
    return c.z * mix(vec3(1.0f), clamp(p - 1.0f, 0.0f, 1.0f), c.y);
}

// Applies gamma using a power curve.
// Small clamps prevent invalid pow() behavior for near-zero values.
//vec3 applyGamma(vec3 c, float gammaValue) {
//    return pow(max(c, vec3(1e-7f)), vec3(max(gammaValue, 0.01f)));
//}

// Applies exposure in photographic stops.
// exp2(x) means +1.0 doubles brightness, -1.0 halves it.
vec3 applyExposure(vec3 c, float exposureValue) {
    return clamp(c * exp2(exposureValue), 0.0f, 1.0f);
}

// Applies a simple additive brightness offset.
//vec3 applyBrightness(vec3 c, float brightValue) {
//    return clamp(c + brightValue, 0.0f, 1.0f);
//}

// Applies contrast around a midpoint of 0.5.
// Values > 1.0 increase contrast, values < 1.0 reduce contrast.
vec3 applyContrast(vec3 c, float contrastValue) {
    return clamp((c - 0.5f) * contrastValue + 0.5f, 0.0f, 1.0f);
}

// Adjusts saturation in HSV space.
// Saturation is scaled, then converted back to RGB.
vec3 applySaturation(vec3 c, float satValue) {
    vec3 hsv = rgb2hsv(c);
    hsv.y = clamp(hsv.y * satValue, 0.0f, 1.0f);
    return clamp(hsv2rgb(hsv), 0.0f, 1.0f);
}

// Rotates hue by the given angle in radians.
// Hue is stored on a wrapped [0, 1) interval.
//vec3 applyHue(vec3 c, float hueRadians) {
//    vec3 hsv = rgb2hsv(c);
//    hsv.x = fract(hsv.x + hueRadians / (2.0f * 3.141592653589793f));
//    return clamp(hsv2rgb(hsv), 0.0f, 1.0f);
//}

// Blends the color toward grayscale based on luminance.
// bwValue = 0.0 keeps color, bwValue = 1.0 produces full grayscale.
//vec3 applyBlackWhite(vec3 c, float bwValue) {
//    float l = rgb2lum(c);
//    return mix(c, vec3(l), clamp(bwValue, 0.0f, 1.0f));
//}

// Applies a simple channel rebalance intended as white balance.
// This is not a standard temperature/tint model.
// It computes a per-pixel balancing factor and blends toward it.
//vec3 applyWhiteBalance(vec3 c, float wbValue) {
//    vec3 balancing = vec3(0.5f / max(c.r, 1e-6f), 0.5f / max(c.g, 1e-6f), 0.5f / max(c.b, 1e-6f));
//
//    float avg = (c.r + c.g + c.b) / 3.0f + 1e-6f;
//    vec3 balanced = clamp(c * balancing * avg, 0.0f, 1.0f);
//
//    return mix(c, balanced, clamp(wbValue, 0.0f, 1.0f));
//}

// Applies a piecewise tone curve across 8 equal input ranges.
// Each segment contributes proportionally within its range.
// This acts like a custom scalar tone remapping applied equally to RGB.
vec3 applyToneCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);

    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);

    // Extract only the portion of c that falls into this segment.
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);

    // Weight that segment and accumulate it.
        total += seg * u_toneCurve[i];
    }

    return clamp(total, 0.0f, 1.0f);
}

// Applies a piecewise per-channel color curve across 8 equal input ranges.
// Unlike the tone curve, each segment uses a vec3 weight so RGB can vary.
vec3 applyColorCurve(vec3 c) {
    const int STEPS = 8;
    vec3 total = vec3(0.0f);
    float stepsF = float(STEPS);

    for(int i = 0; i < STEPS; i++) {
        float fi = float(i);

    // Extract only the portion of c that falls into this segment.
        vec3 seg = clamp(c - fi / stepsF, 0.0f, 1.0f / stepsF);

    // Apply per-channel weighting.
        total += seg * u_colorCurve[i];
    }

    return clamp(total, 0.0f, 1.0f);
}

// Samples the source image and returns RGB only.
vec3 sampleImage(vec2 uv) {
    return texture(u_image, uv).rgb;
}

// Computes Sobel edge magnitude from neighboring pixels.
// This is used as an edge signal for sharpening / enhancement.
vec3 sobelMagnitude(vec2 uv, vec2 texel) {
    vec3 tl = sampleImage(uv + texel * vec2(-1.0f, -1.0f));
    vec3 tc = sampleImage(uv + texel * vec2(0.0f, -1.0f));
    vec3 tr = sampleImage(uv + texel * vec2(1.0f, -1.0f));
    vec3 ml = sampleImage(uv + texel * vec2(-1.0f, 0.0f));
    vec3 mr = sampleImage(uv + texel * vec2(1.0f, 0.0f));
    vec3 bl = sampleImage(uv + texel * vec2(-1.0f, 1.0f));
    vec3 bc = sampleImage(uv + texel * vec2(0.0f, 1.0f));
    vec3 br = sampleImage(uv + texel * vec2(1.0f, 1.0f));

    vec3 gx = -tl - 2.0f * ml - bl + tr + 2.0f * mr + br;
    vec3 gy = -tl - 2.0f * tc - tr + bl + 2.0f * bc + br;

    return sqrt(gx * gx + gy * gy + 1e-7f);
}

// Applies edge-based sharpening.
// This is not classic unsharp masking. Instead, detected edge energy
// is multiplied back into the current color and boosted by sharpValue.
vec3 applySharpen(vec3 c, vec2 uv, vec2 texel, float sharpValue) {
    if(sharpValue <= 0.001f)
        return c;

    vec3 edges = sobelMagnitude(uv, texel);
    return clamp(c + sharpValue * edges * c, 0.0f, 1.0f);
}

// Converts canvas UV coordinates into image UV coordinates while preserving
// image aspect ratio inside the canvas.
//
// Returns false when the current fragment lies outside the displayed image area.
// In that case the caller can output a background color instead.
bool mapCanvasUVToImageUV(vec2 canvasUV, out vec2 imageUV) {
    float canvasAspect = u_canvasSize.x / u_canvasSize.y;
    float imageAspect = u_imageSize.x / u_imageSize.y;

    vec2 scale = vec2(1.0f);

  // Fit the image into the canvas while preserving aspect ratio.
  // One axis may be letterboxed.
    if(imageAspect > canvasAspect) {
        scale.y = canvasAspect / imageAspect;
    } else {
        scale.x = imageAspect / canvasAspect;
    }

  // Recenter and remap the canvas UV into image UV space.
    vec2 uv = (canvasUV - 0.5f) / scale + 0.5f;

  // Reject fragments outside the visible image region.
    if(uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return false;
    }

    imageUV = uv;
    return true;
}

// Processes a single image sample through the full adjustment stack.
// The order matters and directly affects the final look.
vec3 processSample(vec2 uv) {
    vec3 color = sampleImage(uv);

    //color = applyGamma(color, u_gamma);
    color = applySharpen(color, uv, u_texelSize, u_sharp);
    //color = applyWhiteBalance(color, u_wb);
    color = applyExposure(color, u_exposure);
    //color = applyBrightness(color, u_bright);
    color = applyContrast(color, u_contrast);
    color = applySaturation(color, u_saturation);
    //color = applyBlackWhite(color, u_bw);
    color = applyToneCurve(color);
    //color = applyHue(color, u_hue);
    color = applyColorCurve(color);

    return clamp(color, 0.0f, 1.0f);
}

// Applies a small Gaussian-like blur by sampling a 5x5 neighborhood.
// Important: each tap is fully processed first, then averaged.
// That means blur happens over the edited result, not the raw image.
vec3 sampleProcessedBlur(vec2 uv, vec2 texel, float radius) {
    vec3 sum = vec3(0.0f);
    float wsum = 0.0f;

    for(int y = -2; y <= 2; y++) {
        for(int x = -2; x <= 2; x++) {
            vec2 off = vec2(float(x), float(y)) * texel * radius;
            float d2 = float(x * x + y * y);
            float w = exp(-d2 / 4.0f);

            sum += processSample(uv + off) * w;
            wsum += w;
        }
    }

    return sum / max(wsum, 1e-6f);
}

// Main fragment entry point.
void main() {
    vec2 imageUV;

  // Convert canvas-space UV to image-space UV.
  // Output black for pixels outside the fitted image area.
    if(!mapCanvasUVToImageUV(v_uv, imageUV)) {
        outColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    vec3 color;

  // Optional blur path.
    if(u_blur > 0.001f) {
        color = sampleProcessedBlur(imageUV, u_texelSize, u_blur);
    } else {
        color = processSample(imageUV);
    }

    outColor = vec4(clamp(color, 0.0f, 1.0f), 1.0f);
}