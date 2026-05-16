from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from skimage import measure

import social_media_overlay.low_level_image_metrics as impl


# https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py
# Reference implementation copied from the original repository

"""
Tests for the batched PyTorch/Kornia implementation of low-level image metrics.

The production implementation accepts normalized RGB tensors with shape
[B, 3, H, W] and returns one metric value per image. The original reference
implementation accepts image paths, loads images with PIL/OpenCV/skimage, and
returns scalar values for a single image.

These tests serve three purposes:

1. Verify basic implementation behavior, including input validation, expected
   output keys, output shapes, dtype conversion, clamping, and aggregation.

2. Compare the tensor implementation against copied reference functions where
   the metrics are conceptually equivalent, while explicitly documenting the
   expected scale or backend differences. For example, brightness, saturation,
   and RMS contrast are computed on normalized [0, 1] tensors here, whereas the
   reference implementation computes them on uint8 [0, 255] images.

3. Keep direct parity checks against the original implementation visible as
   xfail tests. These tests are expected to fail because the current
   implementation intentionally differs from the reference for LAB conversion,
   value scaling, and blur measurement. If the implementation is later changed
   to reproduce the original behavior exactly, the xfail tests will XPASS and
   should be revisited.

The blur metric is especially important: the current implementation does not
reproduce skimage.measure.blur_effect. It uses Laplacian variance as a
sharpness proxy, so higher values mean sharper images, while the reference blur
metric returns higher values for blurrier images.
"""

def _ref_convert_to_lab(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    np_image = np.array(image)
    return cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)


def _ref_get_lab_channels(lab_image: np.ndarray):
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    return l_channel, a_channel, b_channel


def ref_calculate_colorfulness(image_path: str | Path) -> float:
    image = Image.open(image_path)
    _, a_channel, b_channel = _ref_get_lab_channels(_ref_convert_to_lab(image))

    a_mean, _ = np.mean(a_channel), np.std(a_channel)
    b_mean, _ = np.mean(b_channel), np.std(b_channel)

    a_diff = a_channel - a_mean
    b_diff = b_channel - b_mean
    color_diff = np.sqrt(a_diff ** 2 + b_diff ** 2)
    mean_color_diff = np.mean(color_diff)
    std_color_diff = np.std(color_diff)

    return float(std_color_diff + 0.3 * mean_color_diff)


def ref_compute_mean_brightness(image_path: str | Path) -> float:
    image = Image.open(image_path)
    grayscale_image = image.convert("L")
    np_image = np.array(grayscale_image)
    return float(np.mean(np_image))


def ref_compute_mean_saturation(image_path: str | Path) -> float:
    image = Image.open(image_path).convert("HSV")
    np_image = np.array(image)
    saturation = np_image[:, :, 1]
    return float(np.mean(saturation))


def ref_compute_rms_contrast(image_path: str | Path) -> float:
    image = Image.open(image_path).convert("L")
    np_image = np.array(image)
    return float(np.std(np_image))


def ref_compute_lighting_diversity(image_path: str | Path) -> float:
    image = Image.open(image_path).convert("RGB")
    l_channel, _, _ = _ref_get_lab_channels(_ref_convert_to_lab(image))
    return float(np.std(l_channel))


def ref_compute_blur_effect(image_path: str | Path) -> float:
    image = Image.open(image_path).convert("RGB")
    grayscale_image = np.array(image.convert("L"))
    return float(measure.blur_effect(grayscale_image))


# Shared test infrastructure

def save_tensor_as_png(image_chw: torch.Tensor, path: Path) -> None:
    """
    Save [3, H, W] float image in [0, 1] to PNG so both implementations
    can be evaluated on the exact same visual input.
    """
    image = image_chw.detach().cpu().clamp(0, 1)
    image_hwc = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image_hwc, mode="RGB").save(path)


def to_batch(image_chw: torch.Tensor) -> torch.Tensor:
    return image_chw.unsqueeze(0)


def assert_close_scalar(actual: float, expected: float, atol: float = 1e-6, rtol: float = 1e-6):
    assert math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), (
        f"actual={actual}, expected={expected}, abs_diff={abs(actual - expected)}"
    )


@pytest.fixture
def synthetic_images():
    """
    Deterministic synthetic images covering:
    - constant grayscale
    - grayscale ramp
    - high-frequency checkerboard
    - saturated color blocks
    - random RGB content
    """
    H, W = 32, 32

    gray = torch.full((3, H, W), 0.5, dtype=torch.float32)

    ramp = torch.linspace(0, 1, W, dtype=torch.float32).repeat(H, 1)
    ramp_rgb = torch.stack([ramp, ramp, ramp], dim=0)

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    checker = ((xx + yy) % 2).float()
    checker_rgb = torch.stack([checker, checker, checker], dim=0)

    colors = torch.zeros((3, H, W), dtype=torch.float32)
    colors[:, : H // 2, : W // 2] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    colors[:, : H // 2, W // 2 :] = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1)
    colors[:, H // 2 :, : W // 2] = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
    colors[:, H // 2 :, W // 2 :] = torch.tensor([1.0, 1.0, 0.0]).view(3, 1, 1)

    g = torch.Generator().manual_seed(0)
    random_img = torch.rand((3, H, W), generator=g)

    return {
        "gray": gray,
        "ramp": ramp_rgb,
        "checker": checker_rgb,
        "colors": colors,
        "random": random_img,
    }


# 1. Tests for exact / near-exact behavior that should hold

def test_check_images_rejects_non_tensor():
    with pytest.raises(TypeError):
        impl._check_images(np.zeros((1, 3, 8, 8), dtype=np.float32))


@pytest.mark.parametrize(
    "bad_shape",
    [
        torch.zeros(3, 8, 8),
        torch.zeros(1, 8, 8),
        torch.zeros(1, 1, 8, 8),
        torch.zeros(1, 4, 8, 8),
        torch.zeros(1, 3, 8),
    ],
)
def test_check_images_rejects_bad_shapes(bad_shape):
    with pytest.raises(ValueError):
        impl._check_images(bad_shape)


def test_check_images_converts_non_float_to_float_and_clamps():
    x = torch.tensor(
        [[[
            [-1, 0],
            [1, 2],
        ], [
            [0, 1],
            [2, 3],
        ], [
            [-2, 0],
            [1, 4],
        ]]],
        dtype=torch.int64,
    )
    y = impl._check_images(x)
    assert isinstance(y, torch.Tensor)
    assert torch.is_floating_point(y)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_val_image_properties_returns_expected_keys_and_shapes(synthetic_images):
    batch = torch.stack(list(synthetic_images.values()), dim=0)
    out = impl.val_image_properties(batch)

    expected_keys = {
        "colorfulness",
        "mean_brightness",
        "mean_saturation",
        "rms_contrast",
        "lighting_diversity",
        "blur_effect",
    }
    assert set(out.keys()) == expected_keys

    batch_size = batch.shape[0]
    for key, value in out.items():
        assert isinstance(value, torch.Tensor)
        assert value.shape == (batch_size,), f"{key} has shape {tuple(value.shape)}"


def test_summarize_image_property_dicts_empty():
    out = impl.summarize_image_property_dicts([])
    assert out["num_samples"] == 0
    for key, value in out.items():
        if key != "num_samples":
            assert value == 0.0


def test_summarize_image_property_dicts_nonempty():
    metric_dicts = [
        {
            "colorfulness": torch.tensor([1.0, 3.0]),
            "mean_brightness": torch.tensor([2.0, 4.0]),
            "mean_saturation": torch.tensor([5.0, 7.0]),
            "rms_contrast": torch.tensor([11.0, 13.0]),
            "lighting_diversity": torch.tensor([17.0, 19.0]),
            "blur_effect": torch.tensor([23.0, 29.0]),
        },
        {
            "colorfulness": torch.tensor([5.0]),
            "mean_brightness": torch.tensor([6.0]),
            "mean_saturation": torch.tensor([8.0]),
            "rms_contrast": torch.tensor([15.0]),
            "lighting_diversity": torch.tensor([21.0]),
            "blur_effect": torch.tensor([31.0]),
        },
    ]

    out = impl.summarize_image_property_dicts(metric_dicts)

    assert out["num_samples"] == 3
    assert_close_scalar(out["colorfulness_mean"], np.mean([1.0, 3.0, 5.0]))
    assert_close_scalar(out["colorfulness_std"], np.std([1.0, 3.0, 5.0]))
    assert_close_scalar(out["mean_brightness_mean"], np.mean([2.0, 4.0, 6.0]))
    assert_close_scalar(out["mean_brightness_std"], np.std([2.0, 4.0, 6.0]))
    assert_close_scalar(out["mean_saturation_mean"], np.mean([5.0, 7.0, 8.0]))
    assert_close_scalar(out["mean_saturation_std"], np.std([5.0, 7.0, 8.0]))
    assert_close_scalar(out["rms_contrast_mean"], np.mean([11.0, 13.0, 15.0]))
    assert_close_scalar(out["rms_contrast_std"], np.std([11.0, 13.0, 15.0]))
    assert_close_scalar(out["lighting_diversity_mean"], np.mean([17.0, 19.0, 21.0]))
    assert_close_scalar(out["lighting_diversity_std"], np.std([17.0, 19.0, 21.0]))
    assert_close_scalar(out["blur_effect_mean"], np.mean([23.0, 29.0, 31.0]))
    assert_close_scalar(out["blur_effect_std"], np.std([23.0, 29.0, 31.0]))


# 2.intentional deviations from the original
#
# These tests do NOT expect numerical equality with the reference.
# Instead, they assert the specific, intentional differences.
#
# Intentional deviations in the current implementation:
#
# - mean_brightness:
#     reference returns grayscale mean on uint8/PIL scale [0, 255]
#     implementation returns grayscale mean on normalized tensor scale [0, 1]
#
# - mean_saturation:
#     reference returns HSV saturation on PIL uint8 scale [0, 255]
#     implementation returns HSV saturation on normalized tensor scale [0, 1]
#
# - rms_contrast:
#     reference computes grayscale std on [0, 255]
#     implementation computes grayscale std on [0, 1]
#
# - colorfulness:
#     formula is analogous, but implementation uses Kornia LAB while reference
#     uses OpenCV LAB. Those are not numerically identical representations.
#
# - lighting_diversity:
#     implementation uses Kornia LAB L-channel, reference uses OpenCV LAB L-channel.
#     Different LAB conventions => different numeric outputs.
#
# - blur_effect:
#     implementation intentionally uses Laplacian-variance sharpness proxy.
#     reference uses skimage.measure.blur_effect.
#     These are different metrics, not just different implementations.


@pytest.mark.parametrize("image_id", ["gray", "ramp", "checker", "colors", "random"])
def test_mean_brightness_is_intentionally_normalized(tmp_path: Path, synthetic_images, image_id: str):
    image = synthetic_images[image_id]
    image_path = tmp_path / f"{image_id}.png"
    save_tensor_as_png(image, image_path)

    ref_value = ref_compute_mean_brightness(image_path)
    impl_value = float(impl.compute_mean_brightness(to_batch(image))[0].item())

    # Same conceptual metric, but implementation is normalized to [0, 1].
    assert_close_scalar(impl_value * 255.0, ref_value, atol=1.0, rtol=1e-3)


@pytest.mark.parametrize("image_id", ["gray", "ramp", "checker", "colors", "random"])
def test_mean_saturation_is_intentionally_normalized(tmp_path: Path, synthetic_images, image_id: str):
    image = synthetic_images[image_id]
    image_path = tmp_path / f"{image_id}.png"
    save_tensor_as_png(image, image_path)

    ref_value = ref_compute_mean_saturation(image_path)
    impl_value = float(impl.compute_mean_saturation(to_batch(image))[0].item())

    # Same conceptual metric, but implementation is normalized to [0, 1].
    assert_close_scalar(impl_value * 255.0, ref_value, atol=1.0, rtol=1e-3)


@pytest.mark.parametrize("image_id", ["gray", "ramp", "checker", "colors", "random"])
def test_rms_contrast_is_intentionally_normalized(tmp_path: Path, synthetic_images, image_id: str):
    image = synthetic_images[image_id]
    image_path = tmp_path / f"{image_id}.png"
    save_tensor_as_png(image, image_path)

    ref_value = ref_compute_rms_contrast(image_path)
    impl_value = float(impl.compute_rms_contrast(to_batch(image))[0].item())

    # Same conceptual metric, but implementation is normalized to [0, 1].
    assert_close_scalar(impl_value * 255.0, ref_value, atol=1.0, rtol=1e-3)


@pytest.mark.parametrize("image_id", ["gray", "ramp", "checker", "colors", "random"])
def test_colorfulness_is_not_reference_equivalent_due_to_lab_backend(tmp_path: Path, synthetic_images, image_id: str):
    image = synthetic_images[image_id]
    image_path = tmp_path / f"{image_id}.png"
    save_tensor_as_png(image, image_path)

    ref_value = ref_calculate_colorfulness(image_path)
    impl_value = float(impl.calculate_colorfulness(to_batch(image))[0].item())

    # This test asserts intentional non-equivalence.
    # We do not want exact parity here because Kornia LAB != OpenCV LAB.
    close = math.isclose(impl_value, ref_value, abs_tol=1e-4, rel_tol=1e-4)

    if image_id in {"colors", "random"}:
        assert not close, (
            "Expected colorfulness to differ from the reference because "
            "Kornia LAB and OpenCV LAB are different numeric representations."
        )
    else:
        # Near-grayscale images may still land extremely close to zero in both implementations.
        assert impl_value >= 0.0
        assert ref_value >= 0.0


@pytest.mark.parametrize("image_id", ["gray", "ramp", "checker", "colors", "random"])
def test_lighting_diversity_is_not_reference_equivalent_due_to_lab_backend(tmp_path: Path, synthetic_images, image_id: str):
    image = synthetic_images[image_id]
    image_path = tmp_path / f"{image_id}.png"
    save_tensor_as_png(image, image_path)

    ref_value = ref_compute_lighting_diversity(image_path)
    impl_value = float(impl.compute_lighting_diversity(to_batch(image))[0].item())

    # Intentional non-equivalence because Kornia LAB L-channel != OpenCV LAB L-channel.
    close = math.isclose(impl_value, ref_value, abs_tol=1e-4, rel_tol=1e-4)

    if image_id in {"ramp", "colors", "random"}:
        assert not close, (
            "Expected lighting diversity to differ from the reference because "
            "the implementation uses Kornia LAB whereas the reference uses OpenCV LAB."
        )
    else:
        assert impl_value >= 0.0
        assert ref_value >= 0.0


@pytest.mark.parametrize("image_id", ["gray", "ramp", "checker", "colors", "random"])
def test_blur_effect_is_intentionally_a_different_metric(tmp_path: Path, synthetic_images, image_id: str):
    image = synthetic_images[image_id]
    image_path = tmp_path / f"{image_id}.png"
    save_tensor_as_png(image, image_path)

    ref_value = ref_compute_blur_effect(image_path)
    impl_value = float(impl.compute_blur_effect(to_batch(image))[0].item())

    # The implementation explicitly states that it does NOT reproduce skimage.measure.blur_effect.
    assert not math.isclose(impl_value, ref_value, abs_tol=1e-4, rel_tol=1e-4), (
        "Expected blur metric to differ because implementation uses Laplacian-variance "
        "sharpness proxy instead of skimage.measure.blur_effect."
    )

# 3. Explicit xfail parity tests
#
# These tests encode the original contract directly, but they are marked
# xfail because the current implementation intentionally deviates.
#
# This is useful because:
# - the parity requirement stays visible
# - CI reports these as expected failures instead of regressions
# - if the implementation is later changed to match the original, these
#   tests will start XPASSing and can be revisited

PARITY_CASES = [
    pytest.param(
        "colorfulness",
        ref_calculate_colorfulness,
        impl.calculate_colorfulness,
        2e-3,
        1e-3,
        marks=pytest.mark.xfail(
            reason="Intentional deviation: Kornia LAB does not numerically match OpenCV LAB used by the reference."
        ),
    ),
    pytest.param(
        "mean_brightness",
        ref_compute_mean_brightness,
        impl.compute_mean_brightness,
        1e-3,
        1e-3,
        marks=pytest.mark.xfail(
            reason="Intentional deviation: implementation returns normalized [0,1] brightness, reference returns [0,255]."
        ),
    ),
    pytest.param(
        "mean_saturation",
        ref_compute_mean_saturation,
        impl.compute_mean_saturation,
        1e-3,
        1e-3,
        marks=pytest.mark.xfail(
            reason="Intentional deviation: implementation returns normalized [0,1] saturation, reference returns [0,255]."
        ),
    ),
    pytest.param(
        "rms_contrast",
        ref_compute_rms_contrast,
        impl.compute_rms_contrast,
        1e-3,
        1e-3,
        marks=pytest.mark.xfail(
            reason="Intentional deviation: implementation computes contrast on normalized [0,1] grayscale, reference on [0,255]."
        ),
    ),
    pytest.param(
        "lighting_diversity",
        ref_compute_lighting_diversity,
        impl.compute_lighting_diversity,
        1e-3,
        1e-3,
        marks=pytest.mark.xfail(
            reason="Intentional deviation: Kornia LAB L-channel differs from OpenCV LAB L-channel."
        ),
    ),
    pytest.param(
        "blur_effect",
        ref_compute_blur_effect,
        impl.compute_blur_effect,
        1e-3,
        1e-3,
        marks=pytest.mark.xfail(
            reason="Intentional deviation: implementation uses Laplacian-variance proxy instead of skimage.measure.blur_effect."
        ),
    ),
]


@pytest.mark.parametrize("metric_name, ref_fn, impl_fn, atol, rtol", PARITY_CASES)
def test_metric_matches_original_per_image_xfail(
    tmp_path: Path,
    synthetic_images: dict[str, torch.Tensor],
    metric_name,
    ref_fn,
    impl_fn,
    atol,
    rtol,
):
    """
    Direct parity test against the original implementation.

    Marked xfail because the current implementation intentionally deviates
    from the reference for all listed metrics.
    """
    for image_id, image in synthetic_images.items():
        image_path = tmp_path / f"{metric_name}_{image_id}.png"
        save_tensor_as_png(image, image_path)

        expected = ref_fn(image_path)
        actual = float(impl_fn(to_batch(image))[0].detach().cpu().item())

        assert math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), (
            f"{metric_name} mismatch for {image_id}: "
            f"actual={actual}, expected={expected}"
        )


@pytest.mark.xfail(
    reason=(
        "Intentional deviation: val_image_properties aggregates metrics whose current "
        "implementations do not numerically match the original reference."
    )
)
def test_val_image_properties_matches_original_bundle_xfail(
    tmp_path: Path,
    synthetic_images: dict[str, torch.Tensor],
):
    expected = {
        "colorfulness": [],
        "mean_brightness": [],
        "mean_saturation": [],
        "rms_contrast": [],
        "lighting_diversity": [],
        "blur_effect": [],
    }

    batch = []

    for image_id, image in synthetic_images.items():
        batch.append(image)

        image_path = tmp_path / f"bundle_{image_id}.png"
        save_tensor_as_png(image, image_path)

        expected["colorfulness"].append(ref_calculate_colorfulness(image_path))
        expected["mean_brightness"].append(ref_compute_mean_brightness(image_path))
        expected["mean_saturation"].append(ref_compute_mean_saturation(image_path))
        expected["rms_contrast"].append(ref_compute_rms_contrast(image_path))
        expected["lighting_diversity"].append(ref_compute_lighting_diversity(image_path))
        expected["blur_effect"].append(ref_compute_blur_effect(image_path))

    batch_t = torch.stack(batch, dim=0)
    actual = impl.val_image_properties(batch_t)

    for key, expected_values in expected.items():
        actual_values = actual[key].detach().cpu().numpy()
        expected_values = np.array(expected_values, dtype=np.float64)
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"Mismatch in bundled metric {key}",
        )