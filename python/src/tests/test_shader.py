from __future__ import annotations

"""
These tests compare a **PyTorch translation of the fragment shader** against the
original Kornia/PyTorch reference implementation used by the project.

VERY IMPORTANT
----------------------------------------
The goal here is **not** to prove bit-perfect equality between WebGL shaders and
Kornia.  The goal is to show that the translated shader logic is either

1. essentially identical for simple operations, or
2. acceptably close for operations where the shader and Kornia use slightly
   different implementations.

The most important known and acceptable differences are:

CONTRAST
The shader-style translation uses a 16 x 16 UV-grid approximation of the global
image mean:

.. code-block:: glsl

    float estimateKorniaImageMean() {
        float sum = 0.0;
        const int N = 16;
        for (int y = 0; y < N; y++) {
            float fy = (float(y) + 0.5) / float(N);
            for (int x = 0; x < N; x++) {
                float fx = (float(x) + 0.5) / float(N);
                sum += rgbToGrayscaleKornia(sampleImage(vec2(fx, fy)));
            }
        }
        return sum / float(N * N);
    }

Kornia computes its own internal mean inside
``adjust_contrast_with_mean_subtraction``.  This means we should expect a small
systematic difference even when the shader translation is semantically correct.
That difference is usually tiny and visually irrelevant, so the contrast tests
use a moderate tolerance.

SHARPEN
~~~~~~~
The shader sharpen pass is **not the same algorithm** as
``kornia.enhance.sharpness``.

The shader uses:
- a specific 3 x 3 neighbourhood,
- a center weight of 5,
- division by 13,
- and ``mix(degenerate, c, sharpValue)``.

Kornia's sharpness operator uses its own definition.  Also, for ``sharp = 0``
the shader returns the original image unchanged, while Kornia still returns its
own degenerate base image.  Therefore large numerical differences are expected
here and are not considered important as long as the result still behaves like a
reasonable sharpening operator.

BLUR
~~~~
The shader blur is a separable 7-tap blur with reflected UVs.  The Kornia
reference blur uses ``gaussian_blur2d`` with a larger default kernel.  They are
therefore compared with a tolerance and interpreted as perceptual or
behavioural agreement, not exact equality.

By contrast, exposure and curve adjustments should match almost exactly because
those are straightforward pointwise formulas.
"""

import math

import pytest
import torch

from social_media_overlay.ext.gebhardt.image_transformations.image_transformations import (
    apply_color_curve_adjustment,
    apply_contrast,
    apply_gaussian_blur,
    apply_saturation,
    apply_sharpening,
    apply_tone_curve_adjustment,
)
import social_media_overlay.ext.gebhardt.image_transformations.imgage_transformations_torch_diff as ittf

from helpers_glsl_reference import (
    glsl_apply_blur_native_grid,
    glsl_apply_color_adjustments,
    glsl_apply_contrast,
    glsl_apply_curve_adjustment,
    glsl_apply_exposure,
    glsl_apply_full_pipeline,
    glsl_apply_saturation,
    glsl_apply_sharpen,
)


def assert_equivalent(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_abs_tol: float,
    mean_abs_tol: float,
    min_psnr: float,
) -> None:
    """Assert approximate equality using max error, mean error, and PSNR.

    Why three metrics?
    ------------------
    ``max_abs`` catches single bad pixels, ``mean_abs`` captures the overall
    average drift, and ``PSNR`` provides a familiar signal-to-error measure.
    Using all three makes the tests robust and easy to interpret.
    """
    actual = actual.detach().float()
    expected = expected.detach().float()

    diff = torch.abs(actual - expected)
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    mse = float(torch.mean((actual - expected) ** 2).item())
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * math.log10(1.0 / mse))

    assert max_abs <= max_abs_tol, (
        f"max abs diff {max_abs} > {max_abs_tol}; "
        f"mean abs diff={mean_abs}, psnr={psnr:.2f} dB"
    )
    assert mean_abs <= mean_abs_tol, (
        f"mean abs diff {mean_abs} > {mean_abs_tol}; "
        f"max abs diff={max_abs}, psnr={psnr:.2f} dB"
    )
    assert psnr >= min_psnr, (
        f"psnr {psnr:.2f} dB < {min_psnr} dB; "
        f"max abs diff={max_abs}, mean abs diff={mean_abs}"
    )


# -----------------------------------------------------------------------------
# Test data helpers
# -----------------------------------------------------------------------------

def _rand_chw(seed: int, h: int, w: int) -> torch.Tensor:
    return torch.rand(
        3,
        h,
        w,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float32,
    )


def _constant_chw(value: float, h: int, w: int) -> torch.Tensor:
    return torch.full((3, h, w), value, dtype=torch.float32)


def _impulse_chw(h: int, w: int) -> torch.Tensor:
    img = torch.zeros(3, h, w, dtype=torch.float32)
    img[:, h // 2, w // 2] = 1.0
    return img


def _saturated_chw(h: int, w: int) -> torch.Tensor:
    img = torch.zeros(3, h, w, dtype=torch.float32)
    img[0] = 1.0
    img[1] = 0.15
    img[2] = 0.0
    return img


TEST_IMAGES: dict[str, torch.Tensor] = {
    "black_8x8": _constant_chw(0.0, 8, 8),
    "midgray_8x8": _constant_chw(0.5, 8, 8),
    "impulse_8x8": _impulse_chw(8, 8),
    "saturated_8x8": _saturated_chw(8, 8),
    "random_8x8_seed0": _rand_chw(0, 8, 8),
    "random_32x32_seed0": _rand_chw(0, 32, 32),
    "impulse_32x32": _impulse_chw(32, 32),
}


def _batch(chw: torch.Tensor) -> torch.Tensor:
    return chw.unsqueeze(0).contiguous()


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32)


def _tone_curve(values: list[float]) -> torch.Tensor:
    assert len(values) == 8
    return torch.tensor(values, dtype=torch.float32).view(1, 1, 8, 1)


def _color_curve(values: list[list[float]]) -> torch.Tensor:
    arr = torch.tensor(values, dtype=torch.float32)
    assert arr.shape == (3, 8)
    return arr.view(1, 3, 8, 1)


IDENTITY_TONE = _tone_curve([1.0] * 8)
NONTRIVIAL_TONE = _tone_curve([0.85, 0.95, 1.00, 1.05, 1.10, 1.05, 1.00, 0.95])

IDENTITY_COLOR = _color_curve([
    [1.0] * 8,
    [1.0] * 8,
    [1.0] * 8,
])
NONTRIVIAL_COLOR = _color_curve([
    [0.95, 1.00, 1.02, 1.05, 1.08, 1.08, 1.04, 1.00],
    [1.00, 1.00, 0.98, 0.98, 0.96, 0.96, 0.94, 0.92],
    [1.05, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92],
])


# -----------------------------------------------------------------------------
# Nearly exact tests: exposure and curve operations
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0.0, 0.25, 0.75, 1.5])
@pytest.mark.parametrize("name", ["black_8x8", "midgray_8x8", "random_8x8_seed0"])
def test_shader_exposure_torch_matches_kornia_reference(name: str, value: float) -> None:
    """Exposure should match essentially exactly.

    This operation is just

    ``clamp(image * exp2(exposure), 0, 1)``

    so any meaningful mismatch would likely indicate a real bug.
    """
    img = _batch(TEST_IMAGES[name])
    p = _scalar(value)

    actual = glsl_apply_exposure(img.clone(), p)
    expected = ittf.apply_exposure(img.clone(), p)

    assert_equivalent(actual, expected, max_abs_tol=1e-6, mean_abs_tol=1e-6, min_psnr=100.0)


@pytest.mark.parametrize("curve", [IDENTITY_TONE, NONTRIVIAL_TONE])
@pytest.mark.parametrize("name", ["black_8x8", "midgray_8x8", "random_8x8_seed0"])
def test_shader_tone_curve_torch_matches_kornia_reference(name: str, curve: torch.Tensor) -> None:
    """Tone curve adjustment should match essentially exactly."""
    img = _batch(TEST_IMAGES[name])

    actual = glsl_apply_curve_adjustment(img.clone(), curve)
    expected = apply_tone_curve_adjustment(img.clone(), curve)

    assert_equivalent(actual, expected, max_abs_tol=1e-6, mean_abs_tol=1e-6, min_psnr=100.0)


@pytest.mark.parametrize("curve", [IDENTITY_COLOR, NONTRIVIAL_COLOR])
@pytest.mark.parametrize("name", ["black_8x8", "midgray_8x8", "random_8x8_seed0"])
def test_shader_color_curve_torch_matches_kornia_reference(name: str, curve: torch.Tensor) -> None:
    """Per-channel color curve adjustment should match essentially exactly."""
    img = _batch(TEST_IMAGES[name])

    actual = glsl_apply_curve_adjustment(img.clone(), curve)
    expected = apply_color_curve_adjustment(img.clone(), curve)

    assert_equivalent(actual, expected, max_abs_tol=1e-6, mean_abs_tol=1e-6, min_psnr=100.0)


# -----------------------------------------------------------------------------
# Close but not exact tests: saturation and contrast
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("name", ["midgray_8x8", "saturated_8x8", "random_8x8_seed0"])
def test_shader_saturation_torch_is_close_to_kornia_reference(name: str, value: float) -> None:
    """Saturation should be very close, though tiny HSV differences are okay.

    Why not strict equality?
    ------------------------
    The shader translation uses an explicit HSV conversion that is equivalent in
    meaning but not necessarily identical to Kornia's internal implementation.
    Small floating-point differences here are not important.
    """
    img = _batch(TEST_IMAGES[name])
    p = _scalar(value)

    actual = glsl_apply_saturation(img.clone(), p)
    expected = apply_saturation(img.clone(), p)

    assert_equivalent(actual, expected, max_abs_tol=2e-3, mean_abs_tol=5e-4, min_psnr=55.0)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("name", ["black_8x8", "midgray_8x8", "random_8x8_seed0"])
def test_shader_contrast_torch_is_close_to_kornia_reference(name: str, value: float) -> None:
    """Contrast should be close using the 16 x 16 shader-style mean estimate.

    VERY VISIBLE NOTE
    -----------------
    This test is intentionally *not* strict.  The helper computes the mean with
    the shader-style 16 x 16 sampling approximation, while Kornia computes its
    own internal mean.  Therefore a small stable bias is expected.

    Why this is still okay
    ----------------------
    Both versions still apply contrast around essentially the same global image
    brightness.  The residual difference is very small and is visually not
    meaningful, so a moderate tolerance is the correct test strategy.
    """
    img = _batch(TEST_IMAGES[name])
    p = _scalar(value)

    actual = glsl_apply_contrast(img.clone(), p)
    expected = apply_contrast(img.clone(), p)

    assert_equivalent(actual, expected, max_abs_tol=3.5e-3, mean_abs_tol=3.0e-3, min_psnr=50.0)


# -----------------------------------------------------------------------------
# Known mismatch but acceptable behavioural similarity: sharpen and blur
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0.0, 2.0])
@pytest.mark.parametrize("name", ["impulse_8x8", "random_8x8_seed0"])
def test_shader_sharpen_torch_is_only_loosely_close_to_kornia_reference(name: str, value: float) -> None:
    """Sharpen uses a deliberately wide tolerance.

    VERY VISIBLE NOTE
    -----------------
    The shader sharpen pass and ``kornia.enhance.sharpness`` are different
    algorithms.

    In particular, when ``sharp = 0``:
    - the shader returns the input unchanged,
    - Kornia returns its own degenerate base image.

    That can create large absolute differences, especially on impulse images,
    and this is **expected**.  The tolerance is therefore intentionally wide and
    should be interpreted as "same broad sharpening behaviour" rather than
    "numerically the same operator".

    Why this is still not important
    --------------------------------
    The purpose of the translation check is to ensure the shader helper behaves
    like a plausible sharpening stage in the same family, not to force it to be
    the exact Kornia kernel when the actual shader code is clearly different.
    """
    img = _batch(TEST_IMAGES[name])
    p = _scalar(value)

    actual = glsl_apply_sharpen(img.clone(), p)
    expected = apply_sharpening(img.clone(), p)

    assert_equivalent(actual, expected, max_abs_tol=0.65, mean_abs_tol=0.10, min_psnr=15.0)


@pytest.mark.parametrize("value", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("name", ["impulse_32x32", "random_32x32_seed0"])
def test_shader_blur_torch_is_only_loosely_close_to_kornia_reference(name: str, value: float) -> None:
    """Blur uses a generous tolerance because the kernels are different.

    VERY VISIBLE NOTE
    -----------------
    The shader blur is a separable 7-tap blur with reflected UVs.
    Kornia's reference blur uses ``gaussian_blur2d`` with its own larger kernel.

    Therefore exact equality would be the wrong expectation.  We only test that
    the shader-style translation lands in a broadly similar smoothing regime.
    """
    img = _batch(TEST_IMAGES[name])
    p = _scalar(value)

    actual = glsl_apply_blur_native_grid(img.clone(), p)
    expected = apply_gaussian_blur(img.clone(), p)

    assert_equivalent(actual, expected, max_abs_tol=0.20, mean_abs_tol=0.03, min_psnr=20.0)


# -----------------------------------------------------------------------------
# Pipeline tests
# -----------------------------------------------------------------------------

def test_shader_color_pipeline_torch_is_close_to_kornia_reference() -> None:
    """The full color pass should be close to the sequential Kornia reference.

    This includes the only slightly inexact contrast stage, so the tolerance is
    looser than the exposure/curve tests but still fairly tight.
    """
    img = _batch(TEST_IMAGES["random_8x8_seed0"])
    exposure = _scalar(0.35)
    saturation = _scalar(1.20)
    contrast = _scalar(0.80)

    actual = glsl_apply_color_adjustments(
        img.clone(),
        exposure=exposure,
        saturation=saturation,
        tone=NONTRIVIAL_TONE,
        color=NONTRIVIAL_COLOR,
        contrast=contrast,
    )

    expected = img.clone()
    expected = ittf.apply_exposure(expected, exposure)
    expected = apply_saturation(expected, saturation)
    expected = apply_tone_curve_adjustment(expected, NONTRIVIAL_TONE)
    expected = apply_color_curve_adjustment(expected, NONTRIVIAL_COLOR)
    expected = apply_contrast(expected, contrast)

    assert_equivalent(actual, expected, max_abs_tol=4e-3, mean_abs_tol=2e-3, min_psnr=48.0)


def test_shader_color_then_sharp_pipeline_torch_is_acceptably_close_to_kornia_reference() -> None:
    """Color + sharpen pipeline should stay acceptably close overall.

    VERY VISIBLE NOTE
    -----------------
    Even though sharpening alone is a known non-identical stage, in a realistic
    pipeline with moderate parameters the overall difference remains small.
    That is why this pipeline test can still use a tighter tolerance than the
    standalone sharpen stress test above.
    """
    img = _batch(TEST_IMAGES["random_8x8_seed0"])
    exposure = _scalar(0.20)
    saturation = _scalar(1.10)
    contrast = _scalar(0.90)
    sharp = _scalar(1.15)

    actual = glsl_apply_full_pipeline(
        img.clone(),
        exposure=exposure,
        saturation=saturation,
        tone=NONTRIVIAL_TONE,
        color=NONTRIVIAL_COLOR,
        contrast=contrast,
        sharp=sharp,
        blur=None,
    )

    expected = img.clone()
    expected = ittf.apply_exposure(expected, exposure)
    expected = apply_saturation(expected, saturation)
    expected = apply_tone_curve_adjustment(expected, NONTRIVIAL_TONE)
    expected = apply_color_curve_adjustment(expected, NONTRIVIAL_COLOR)
    expected = apply_contrast(expected, contrast)
    expected = apply_sharpening(expected, sharp)

    assert_equivalent(actual, expected, max_abs_tol=2.5e-2, mean_abs_tol=1e-3, min_psnr=55.0)


def test_shader_full_pipeline_with_blur_torch_is_reasonably_close_to_kornia_reference() -> None:
    """Full pipeline including blur should remain in the same visual regime.

    VERY VISIBLE NOTE
    -----------------
    This is the least strict test in spirit because it combines all known small
    approximations plus the known blur-kernel mismatch.  A moderate tolerance is
    therefore intentional and correct.
    """
    img = _batch(TEST_IMAGES["random_32x32_seed0"])
    exposure = _scalar(0.25)
    saturation = _scalar(1.10)
    contrast = _scalar(0.85)
    sharp = _scalar(1.05)
    blur = _scalar(1.00)

    actual = glsl_apply_full_pipeline(
        img.clone(),
        exposure=exposure,
        saturation=saturation,
        tone=NONTRIVIAL_TONE,
        color=NONTRIVIAL_COLOR,
        contrast=contrast,
        sharp=sharp,
        blur=blur,
    )

    expected = img.clone()
    expected = ittf.apply_exposure(expected, exposure)
    expected = apply_saturation(expected, saturation)
    expected = apply_tone_curve_adjustment(expected, NONTRIVIAL_TONE)
    expected = apply_color_curve_adjustment(expected, NONTRIVIAL_COLOR)
    expected = apply_contrast(expected, contrast)
    expected = apply_sharpening(expected, sharp)
    expected = apply_gaussian_blur(expected, blur)

    assert_equivalent(actual, expected, max_abs_tol=0.20, mean_abs_tol=0.03, min_psnr=20.0)