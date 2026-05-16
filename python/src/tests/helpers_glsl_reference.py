from __future__ import annotations

"""
helpers_glsl_reference.py
=========================

This module contains a **PyTorch translation of the fragment shaders** so that
unit tests can compare the shader-style implementation against the original
Kornia/PyTorch reference implementation.

IMPORTANT TESTING INTENT
------------------------
These helpers are **not** intended to prove bit-identical GPU output.  They are
intended to model the *mathematical meaning* of the shader code as closely as is
reasonable in PyTorch.

The most important consequence is that some operations are only expected to be
**approximately equal** to the Kornia reference:

1. Contrast
   The shader estimates a grayscale image mean on a fixed 16 x 16 UV grid.
   Kornia computes its own internal mean inside
   ``adjust_contrast_with_mean_subtraction``.  Both are conceptually the same
   operation, but they are not numerically identical.

2. Sharpen
   The shader sharpen pass is not the same implementation as
   ``kornia.enhance.sharpness``.  The shader uses its own 3 x 3 weighted
   neighbourhood and a GLSL ``mix`` style interpolation, while Kornia uses its
   own sharpness definition.  They are visually similar edge-enhancement
   operations, but not exact matches.

3. Blur
   The shader blur is a separable 7-tap Gaussian-like blur with UV reflection.
   The Python reference uses Kornia's ``gaussian_blur2d`` with a larger default
   kernel.  Again, the result is similar but not bit-identical.

For exposure and curve adjustments we expect nearly exact agreement because
those are simple pointwise formulas.
"""

import torch


def _as_broadcastable_scalar(x: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
    """Reshape a scalar or 1D parameter so it broadcasts like a GLSL uniform.

    GLSL uniforms such as ``u_exposure`` or ``u_contrast`` are scalar values
    that apply to every fragment.  In PyTorch tests we often store them as
    ``tensor([value])``.  This helper inserts singleton dimensions until the
    tensor broadcasts against a reference image tensor.
    """
    x = torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    while x.ndim < ref.ndim:
        x = x[..., None]
    return x


def center_uv_grid(h: int, w: int, *, device, dtype) -> torch.Tensor:
    """Return pixel-centre UV coordinates with shape ``1 x H x W x 2``.

    Equivalent shader idea
    ----------------------
    In the fullscreen render, every fragment conceptually corresponds to a pixel
    centre.  The matching normalized UV for pixel ``(x, y)`` is:

    ``u = (x + 0.5) / width``
    ``v = (y + 0.5) / height``

    Python equivalent
    -----------------
    We build that same UV grid explicitly so later helper functions can mimic
    ``texture(u_image, v_uv)`` on the discrete image tensor.
    """
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / float(h)
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / float(w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)[None, ...]


def glsl_sample_nearest(image_bchw: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour sampling with clamp-to-edge semantics.

    Equivalent shader definition
    ----------------------------
    The renderer uploads textures with ``NEAREST`` filtering and
    ``CLAMP_TO_EDGE`` wrapping.  Therefore:

    ``sampleImage(uv) = texture(u_image, uv).rgb``

    behaves like a nearest-neighbour lookup on the normalized UV domain.

    Python equivalent
    -----------------
    For each UV coordinate we compute the texel index with:

    ``x = floor(u * width)``
    ``y = floor(v * height)``

    and clamp the indices to the valid image range.

    Parameters
    ----------
    image_bchw:
        Image tensor in ``B x C x H x W`` format.
    uv:
        Either ``B x H x W x 2`` for an image-sized grid or ``B x 2`` / ``2``
        for point samples.
    """
    b, c, h, w = image_bchw.shape

    if uv.ndim == 1:
        uv = uv[None, :].expand(b, -1)

    if uv.ndim == 2:
        x = torch.floor(uv[:, 0] * w).long().clamp(0, w - 1)
        y = torch.floor(uv[:, 1] * h).long().clamp(0, h - 1)
        batch_idx = torch.arange(b, device=image_bchw.device)
        return image_bchw[batch_idx, :, y, x]

    if uv.ndim != 4:
        raise ValueError(f"Unsupported uv shape {tuple(uv.shape)}")

    if uv.shape[0] == 1 and b != 1:
        uv = uv.expand(b, -1, -1, -1)

    x = torch.floor(uv[..., 0] * w).long().clamp(0, w - 1)
    y = torch.floor(uv[..., 1] * h).long().clamp(0, h - 1)
    batch_idx = torch.arange(b, device=image_bchw.device).view(b, 1, 1).expand_as(x)

    sampled = image_bchw[batch_idx, :, y, x]  # B,H,W,C
    return sampled.permute(0, 3, 1, 2).contiguous()


def glsl_rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB to HSV, matching the shader's ``rgb2hsv`` semantics.

    Equivalent shader definition
    ----------------------------
    The fragment shader computes HSV using a branchless ``mix`` / ``step``
    formulation:

    .. code-block:: glsl

        vec3 rgb2hsv(vec3 c) {
            vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
            vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
            vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
            float d = q.x - min(q.w, q.y);
            float e = 1.0e-10;
            return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
        }

    Python equivalent
    -----------------
    We implement the mathematically equivalent HSV conversion in explicit tensor
    form:

    - ``value = max(rgb)``
    - ``delta = max(rgb) - min(rgb)``
    - ``saturation = delta / (value + eps)``
    - ``hue`` is determined by the channel that attains the maximum.

    This is easier to audit than the branchless GLSL form while representing the
    same HSV transform.
    """
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    cmax, cmax_idx = image.max(dim=1, keepdim=True)
    cmin = image.min(dim=1, keepdim=True).values
    delta = cmax - cmin

    eps = torch.tensor(1.0e-10, dtype=image.dtype, device=image.device)
    s = delta / (cmax + eps)

    h = torch.zeros_like(cmax)
    nonzero = delta > 0

    h_r = ((g - b) / (delta + eps)) % 6.0
    h_g = ((b - r) / (delta + eps)) + 2.0
    h_b = ((r - g) / (delta + eps)) + 4.0

    h = torch.where((cmax_idx == 0) & nonzero, h_r, h)
    h = torch.where((cmax_idx == 1) & nonzero, h_g, h)
    h = torch.where((cmax_idx == 2) & nonzero, h_b, h)
    h = (h / 6.0) % 1.0

    return torch.cat([h, s, cmax], dim=1)


def glsl_hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert HSV back to RGB, matching the shader's ``hsv2rgb`` semantics.

    Equivalent shader definition
    ----------------------------
    The shader uses the compact formula:

    .. code-block:: glsl

        vec3 hsv2rgb(vec3 c) {
            vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
            return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
        }

    Python equivalent
    -----------------
    We implement the standard six-sector HSV-to-RGB conversion.  This is again
    mathematically equivalent, just written in a form that is easier to test.
    """
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    hi = torch.floor(h * 6.0).to(torch.int64) % 6
    f = (h * 6.0) - torch.floor(h * 6.0)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    out = torch.empty_like(hsv)
    out = torch.where((hi == 0).expand_as(out), torch.cat([v, t, p], dim=1), out)
    out = torch.where((hi == 1).expand_as(out), torch.cat([q, v, p], dim=1), out)
    out = torch.where((hi == 2).expand_as(out), torch.cat([p, v, t], dim=1), out)
    out = torch.where((hi == 3).expand_as(out), torch.cat([p, q, v], dim=1), out)
    out = torch.where((hi == 4).expand_as(out), torch.cat([t, p, v], dim=1), out)
    out = torch.where((hi == 5).expand_as(out), torch.cat([v, p, q], dim=1), out)
    return out


def glsl_apply_exposure(image: torch.Tensor, exposure: torch.Tensor | float) -> torch.Tensor:
    """Apply exposure exactly like the shader.

    Equivalent shader definition
    ----------------------------
    .. code-block:: glsl

        vec3 applyExposure(vec3 c, float exposureValue) {
            return clamp(c * exp2(exposureValue), 0.0, 1.0);
        }

    Python equivalent
    -----------------
    ``torch.exp2`` matches GLSL ``exp2``, and ``torch.clamp`` matches GLSL
    ``clamp``.
    """
    exposure = _as_broadcastable_scalar(exposure, image)
    return torch.clamp(image * torch.exp2(exposure), 0.0, 1.0)


def glsl_apply_saturation(image: torch.Tensor, saturation: torch.Tensor | float) -> torch.Tensor:
    """Apply saturation exactly like the shader.

    Equivalent shader definition
    ----------------------------
    .. code-block:: glsl

        vec3 applySaturation(vec3 c, float satValue) {
            vec3 hsv = rgb2hsv(c);
            hsv.y = clamp(hsv.y * satValue, 0.0, 1.0);
            return clamp(hsv2rgb(hsv), 0.0, 1.0);
        }

    Python equivalent
    -----------------
    Convert to HSV, multiply the saturation channel, clamp it to ``[0, 1]``,
    convert back to RGB, and clamp the output RGB.
    """
    saturation = _as_broadcastable_scalar(saturation, image[:, :1])
    hsv = glsl_rgb_to_hsv(image)
    hsv[:, 1:2] = torch.clamp(hsv[:, 1:2] * saturation, 0.0, 1.0)
    return torch.clamp(glsl_hsv_to_rgb(hsv), 0.0, 1.0)


def glsl_apply_curve_adjustment(image: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Apply tone or color curve adjustment exactly like the shader.

    Equivalent shader definition
    ----------------------------
    The tone and color curve shaders both perform the same piecewise-linear
    accumulation over eight segments:

    .. code-block:: glsl

        const int STEPS = 8;
        vec3 total = vec3(0.0);
        for (int i = 0; i < STEPS; ++i) {
            vec3 seg = clamp(c - float(i) / float(STEPS), 0.0, 1.0 / float(STEPS));
            total += seg * curve[i];
        }
        return clamp(total, 0.0, 1.0);

    Python equivalent
    -----------------
    We loop over the same eight segments, compute the same clamped segment, and
    accumulate it with the corresponding per-segment weight.

    ``param`` shapes
    ----------------
    - Tone curve: ``B x 1 x 8 x 1``
    - Color curve: ``B x 3 x 8 x 1``

    The tone curve broadcasts over RGB channels exactly like multiplying a GLSL
    ``float`` with a ``vec3``.
    """
    steps = int(param.shape[2])
    out = torch.zeros_like(image)
    for i in range(steps):
        seg = torch.clamp(
            image - (float(i) / float(steps)),
            min=0.0,
            max=1.0 / float(steps),
        )
        weight = param[:, :, i, 0].unsqueeze(-1).unsqueeze(-1)
        out = out + seg * weight
    return torch.clamp(out, 0.0, 1.0)


def glsl_estimate_kornia_image_mean(source_bchw: torch.Tensor, grid_n: int = 16) -> torch.Tensor:
    """Approximate the shader's Kornia-style grayscale image mean.

    Equivalent shader definition
    ----------------------------
    The shader-style approximation used by the tests is:

    .. code-block:: glsl

        float estimateKorniaImageMean() {
            float sum = 0.0;
            const int N = 16;
            for (int y = 0; y < N; y++) {
                float fy = (float(y) + 0.5) / float(N);
                for (int x = 0; x < N; x++) {
                    float fx = (float(x) + 0.5) / float(N);
                    sum += 0.299 * sample.r + 0.587 * sample.g + 0.114 * sample.b;
                }
            }
            return sum / float(N * N);
        }

    Python equivalent
    -----------------
    We sample the image on the same 16 x 16 grid in UV space using the same
    nearest-neighbour sampler used elsewhere in the shader reference, convert
    each sample to grayscale with the same coefficients,

    ``gray = 0.299 * r + 0.587 * g + 0.114 * b``

    and average over all sampled positions.

    WHY THIS IS ONLY AN APPROXIMATION
    ---------------------------------
    Kornia's internal mean computation is not necessarily implemented as this
    exact 16 x 16 sampling grid, so a **small but consistent** mismatch is
    expected.  This is not important for the test goal because both versions are
    still applying contrast around essentially the same global image brightness.
    """
    device = source_bchw.device
    dtype = source_bchw.dtype
    ys = (torch.arange(grid_n, device=device, dtype=dtype) + 0.5) / float(grid_n)
    xs = (torch.arange(grid_n, device=device, dtype=dtype) + 0.5) / float(grid_n)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack([xx, yy], dim=-1)[None, ...].expand(source_bchw.shape[0], -1, -1, -1)

    sampled = glsl_sample_nearest(source_bchw, uv)
    gray = 0.299 * sampled[:, 0:1] + 0.587 * sampled[:, 1:2] + 0.114 * sampled[:, 2:3]
    return gray.mean(dim=(-2, -1), keepdim=True)


def glsl_apply_contrast(
    current_bchw: torch.Tensor,
    contrast: torch.Tensor | float,
    source_for_mean_bchw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply contrast using the shader's mean-estimation idea.

    Equivalent shader definition
    ----------------------------
    .. code-block:: glsl

        vec3 applyContrast(vec3 c, float contrastValue, float imgMean) {
            return clamp(
                c * contrastValue + vec3(imgMean) * (1.0 - contrastValue),
                0.0,
                1.0
            );
        }

    Python equivalent
    -----------------
    1. Estimate a scalar grayscale image mean with
       ``glsl_estimate_kornia_image_mean``.
    2. Apply the same affine interpolation between the current colour and that
       mean value.

    WHY THIS IS ONLY APPROXIMATELY EQUAL TO KORNIA
    ----------------------------------------------
    Kornia's ``adjust_contrast_with_mean_subtraction`` computes its own internal
    mean, whereas the shader reference uses the explicit 16 x 16 approximation
    above.  The resulting difference is typically very small, predictable, and
    visually negligible, so the unit test uses a moderate tolerance instead of a
    bit-exact tolerance.
    """
    if source_for_mean_bchw is None:
        source_for_mean_bchw = current_bchw
    contrast = _as_broadcastable_scalar(contrast, current_bchw)
    img_mean = glsl_estimate_kornia_image_mean(source_for_mean_bchw)
    return torch.clamp(current_bchw * contrast + img_mean * (1.0 - contrast), 0.0, 1.0)


def glsl_apply_color_adjustments(
    image_bchw: torch.Tensor,
    *,
    exposure: torch.Tensor | float,
    saturation: torch.Tensor | float,
    tone: torch.Tensor,
    color: torch.Tensor,
    contrast: torch.Tensor | float,
) -> torch.Tensor:
    """Apply the shader color pass in the same order as the fragment shader.

    Equivalent shader order
    -----------------------
    .. code-block:: glsl

        color = applyExposure(color, u_exposure);
        color = applySaturation(color, u_saturation);
        color = applyToneCurve(color);
        color = applyColorCurve(color);
        color = applyContrast(color, u_contrast, imgMean);

    Python equivalent
    -----------------
    We apply the same operations in the same sequence.  This matters because
    these operations are not commutative.
    """
    out = glsl_apply_exposure(image_bchw, exposure)
    out = glsl_apply_saturation(out, saturation)
    out = glsl_apply_curve_adjustment(out, tone)
    out = glsl_apply_curve_adjustment(out, color)
    out = glsl_apply_contrast(out, contrast)
    return out


def glsl_apply_sharpen(image_bchw: torch.Tensor, sharp: torch.Tensor | float) -> torch.Tensor:
    """Apply the shader sharpen pass.

    Equivalent shader definition
    ----------------------------
    The fragment shader performs:

    .. code-block:: glsl

        if (sharpValue <= 0.001) return c;
        if (border pixel) return c;

        vec3 sum =
            n00 + n10 + n20 +
            n01 + 5.0 * c + n21 +
            n02 + n12 + n22;

        vec3 degenerate = clamp(sum * (1.0 / 13.0), 0.0, 1.0);
        return clamp(mix(degenerate, c, sharpValue), 0.0, 1.0);

    Python equivalent
    -----------------
    - For border pixels we keep the input unchanged.
    - For interior pixels we compute the same eight-neighbour sum plus
      ``5 * centre``.
    - We divide by ``13`` to obtain the "degenerate" base image.
    - We reproduce GLSL ``mix(degenerate, c, sharpValue)`` as
      ``degenerate * (1 - sharpValue) + c * sharpValue``.

    WHY THIS IS ONLY APPROXIMATELY EQUAL TO KORNIA
    ----------------------------------------------
    Kornia's ``sharpness`` uses a different sharpening definition.  The shader's
    behaviour at ``sharp = 0`` is especially different: the shader returns the
    original image unchanged, while Kornia's implementation still returns its
    own degenerate base image.  That creates a large numerical difference which
    is expected and documented by the test tolerances.
    """
    sharp_t = _as_broadcastable_scalar(sharp, image_bchw[:, :1, :, :])
    out = image_bchw.clone()

    # If sharpening is disabled, the shader returns the input unchanged.
    disabled_batches = (sharp_t[:, :, 0:1, 0:1] <= 0.001).view(-1)
    if bool(disabled_batches.all()):
        return out

    if image_bchw.shape[-2] < 3 or image_bchw.shape[-1] < 3:
        return out

    c = image_bchw[:, :, 1:-1, 1:-1]
    sum_ = (
        image_bchw[:, :, :-2, :-2]
        + image_bchw[:, :, :-2, 1:-1]
        + image_bchw[:, :, :-2, 2:]
        + image_bchw[:, :, 1:-1, :-2]
        + 5.0 * c
        + image_bchw[:, :, 1:-1, 2:]
        + image_bchw[:, :, 2:, :-2]
        + image_bchw[:, :, 2:, 1:-1]
        + image_bchw[:, :, 2:, 2:]
    )
    degenerate = torch.clamp(sum_ * (1.0 / 13.0), 0.0, 1.0)
    sharp_inner = sharp_t.expand_as(image_bchw)[:, :, 1:-1, 1:-1]
    mixed = degenerate * (1.0 - sharp_inner) + c * sharp_inner
    out[:, :, 1:-1, 1:-1] = torch.clamp(mixed, 0.0, 1.0)

    if bool(disabled_batches.any()):
        out[disabled_batches] = image_bchw[disabled_batches]

    return out


def reflect_uv(uv: torch.Tensor) -> torch.Tensor:
    """Reflect UV coordinates into the unit square exactly like the shader.

    Equivalent shader definition
    ----------------------------
    .. code-block:: glsl

        vec2 reflectUV(vec2 uv) {
            uv = mod(uv, 2.0);
            uv = abs(uv);
            return 1.0 - abs(1.0 - uv);
        }

    Python equivalent
    -----------------
    ``torch.remainder`` implements the ``mod`` behaviour needed here.
    """
    uv = torch.remainder(uv, 2.0)
    uv = torch.abs(uv)
    return 1.0 - torch.abs(1.0 - uv)


def _gaussian1d(x: int, sigma: torch.Tensor) -> torch.Tensor:
    """Compute the same unnormalized 1D Gaussian weight as the shader."""
    sigma = torch.clamp(sigma, min=1.0e-6)
    x_t = torch.as_tensor(float(x), dtype=sigma.dtype, device=sigma.device)
    return torch.exp(-(x_t * x_t) / (2.0 * sigma * sigma))


def glsl_apply_blur_native_grid(image_bchw: torch.Tensor, blur: torch.Tensor | float) -> torch.Tensor:
    """Apply the shader's separable 7-tap blur.

    Equivalent shader definition
    ----------------------------
    The shader performs a horizontal 7-tap pass over offsets ``-3 .. 3`` and
    then a vertical 7-tap pass over offsets ``-3 .. 3``.  Each pass uses
    ``gaussian1D(offset, sigma)``, reflects UVs with ``reflectUV``, samples the
    intermediate texture, accumulates weighted colours, divides by the weight
    sum, and clamps the result.

    Python equivalent
    -----------------
    We reproduce the same two-pass structure on the tensor grid using the same
    centre UV grid, the same reflected UV coordinates, and the same nearest
    sampler.

    WHY THIS IS ONLY APPROXIMATELY EQUAL TO KORNIA
    ----------------------------------------------
    Kornia's reference blur uses ``gaussian_blur2d`` with a larger default
    kernel.  The shader blur and the Kornia blur both smooth the image, but they
    are not the same kernel, so tests use a generous tolerance.
    """
    b, _c, h, w = image_bchw.shape
    uv = center_uv_grid(h, w, device=image_bchw.device, dtype=image_bchw.dtype).expand(b, -1, -1, -1)
    texel = torch.tensor([1.0 / w, 1.0 / h], dtype=image_bchw.dtype, device=image_bchw.device).view(1, 1, 1, 2)
    sigma = _as_broadcastable_scalar(blur, image_bchw[:, :1, :1, :1])

    # Horizontal pass.
    hsum = torch.zeros_like(image_bchw)
    hwsum = torch.zeros_like(image_bchw[:, :1])
    for dx in range(-3, 4):
        weight = _gaussian1d(dx, sigma)
        offset = torch.tensor([float(dx), 0.0], dtype=image_bchw.dtype, device=image_bchw.device).view(1, 1, 1, 2)
        suv = reflect_uv(uv + offset * texel)
        sample = glsl_sample_nearest(image_bchw, suv)
        hsum = hsum + sample * weight
        hwsum = hwsum + weight
    tmp = torch.clamp(hsum / torch.clamp(hwsum, min=1.0e-6), 0.0, 1.0)

    # Vertical pass.
    vsum = torch.zeros_like(image_bchw)
    vwsum = torch.zeros_like(image_bchw[:, :1])
    for dy in range(-3, 4):
        weight = _gaussian1d(dy, sigma)
        offset = torch.tensor([0.0, float(dy)], dtype=image_bchw.dtype, device=image_bchw.device).view(1, 1, 1, 2)
        suv = reflect_uv(uv + offset * texel)
        sample = glsl_sample_nearest(tmp, suv)
        vsum = vsum + sample * weight
        vwsum = vwsum + weight
    return torch.clamp(vsum / torch.clamp(vwsum, min=1.0e-6), 0.0, 1.0)


def glsl_apply_full_pipeline(
    image_bchw: torch.Tensor,
    *,
    exposure: torch.Tensor | float,
    saturation: torch.Tensor | float,
    tone: torch.Tensor,
    color: torch.Tensor,
    contrast: torch.Tensor | float,
    sharp: torch.Tensor | float,
    blur: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Apply the full shader-style pipeline in the same high-level order.

    The order is:

    1. exposure
    2. saturation
    3. tone curve
    4. color curve
    5. contrast
    6. sharpen
    7. blur (optional)
    """
    out = glsl_apply_color_adjustments(
        image_bchw,
        exposure=exposure,
        saturation=saturation,
        tone=tone,
        color=color,
        contrast=contrast,
    )
    out = glsl_apply_sharpen(out, sharp)
    if blur is not None:
        out = glsl_apply_blur_native_grid(out, blur)
    return torch.clamp(out, 0.0, 1.0)