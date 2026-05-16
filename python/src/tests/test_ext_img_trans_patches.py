import torch
from social_media_overlay.ext.gebhardt.image_transformations.imgage_transformations_torch_diff import apply_curve_adjustment_patch


import torch
from social_media_overlay.ext.gebhardt.image_transformations.imgage_transformations_torch_diff import (
    apply_curve_adjustment_patch,
)


def apply_curve_adjustment_original(im, param, normalize=False):
    curve_steps = param.shape[2]
    total_image = im * 0
    param_list = torch.split(param, 1, dim=2)

    for i in range(curve_steps):
        total_image += torch.clamp(im - 1.0 * i / curve_steps, 0, 1.0 / curve_steps) * param_list[i]

    if normalize:
        color_curve_sum = param.sum(dim=2, keepdim=True) + 1e-9
        total_image *= curve_steps / color_curve_sum
    else:
        total_image = torch.clamp(total_image, max=1.0)

    return total_image


def test_apply_curve_adjustment_patch_shape():
    im = torch.rand(2, 3, 1, 4, 4)
    param = torch.rand(2, 3, 5, 4, 4)

    out = apply_curve_adjustment_patch(im, param)

    assert out.shape == im.shape


def test_apply_curve_adjustment_patch_clamped_output():
    im = torch.rand(1, 3, 1, 4, 4) * 2
    param = torch.ones(1, 3, 4, 4, 4)

    out = apply_curve_adjustment_patch(im, param, normalize=False)

    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_apply_curve_adjustment_patch_zero_param():
    im = torch.rand(1, 3, 1, 4, 4)
    param = torch.zeros(1, 3, 4, 4, 4)

    out = apply_curve_adjustment_patch(im, param)

    assert torch.allclose(out, torch.zeros_like(im))


def test_apply_curve_adjustment_patch_normalize_changes_result():
    im = torch.rand(1, 3, 1, 4, 4)
    param = torch.rand(1, 3, 4, 4, 4)

    out_no_norm = apply_curve_adjustment_patch(im, param, normalize=False)
    out_norm = apply_curve_adjustment_patch(im, param, normalize=True)

    assert not torch.allclose(out_no_norm, out_norm)


def test_apply_curve_adjustment_patch_simple_case():
    im = torch.ones(1, 1, 1, 1, 1)
    param = torch.ones(1, 1, 2, 1, 1)

    out_original = apply_curve_adjustment_original(im, param, normalize=False)
    out_patch = apply_curve_adjustment_patch(im, param, normalize=False)

    assert torch.allclose(out_patch, out_original, atol=1e-6, rtol=1e-6)
    assert torch.isclose(out_patch[0, 0, 0, 0, 0], torch.tensor(1.0))


def test_apply_curve_adjustment_patch_normalization_behavior():
    im = torch.ones(1, 1, 1, 1, 1)
    param = torch.ones(1, 1, 2, 1, 1)

    out = apply_curve_adjustment_patch(im, param, normalize=True)

    assert torch.isfinite(out).all()
    assert out.item() > 0


def test_apply_curve_adjustment_patch_matches_original_without_normalize():
    torch.manual_seed(0)

    im = torch.rand(2, 3, 1, 4, 4)
    param = torch.rand(2, 3, 5, 4, 4)

    out_original = apply_curve_adjustment_original(im, param, normalize=False)
    out_patch = apply_curve_adjustment_patch(im, param, normalize=False)

    assert torch.allclose(out_patch, out_original, atol=1e-6, rtol=1e-6)


def test_apply_curve_adjustment_patch_matches_original_with_normalize():
    torch.manual_seed(0)

    im = torch.rand(2, 3, 1, 4, 4)
    param = torch.rand(2, 3, 5, 4, 4)

    out_original = apply_curve_adjustment_original(im, param, normalize=True)
    out_patch = apply_curve_adjustment_patch(im, param, normalize=True)

    assert torch.allclose(out_patch, out_original, atol=1e-6, rtol=1e-6)


def test_apply_curve_adjustment_patch_matches_original_across_multiple_inputs():
    for seed in range(5):
        torch.manual_seed(seed)

        im = torch.rand(2, 3, 1, 4, 4)
        param = torch.rand(2, 3, 4, 4, 4)

        out_original = apply_curve_adjustment_original(im, param, normalize=False)
        out_patch = apply_curve_adjustment_patch(im, param, normalize=False)

        assert torch.allclose(out_patch, out_original, atol=1e-6, rtol=1e-6)


def test_apply_curve_adjustment_patch_matches_original_across_multiple_inputs_normalized():
    for seed in range(5):
        torch.manual_seed(seed)

        im = torch.rand(2, 3, 1, 4, 4)
        param = torch.rand(2, 3, 4, 4, 4)

        out_original = apply_curve_adjustment_original(im, param, normalize=True)
        out_patch = apply_curve_adjustment_patch(im, param, normalize=True)

        assert torch.allclose(out_patch, out_original, atol=1e-6, rtol=1e-6)


def test_apply_curve_adjustment_patch_preserves_shape():
    im = torch.rand(1, 3, 1, 8, 8)
    param = torch.rand(1, 3, 6, 8, 8)

    out = apply_curve_adjustment_patch(im, param)

    assert out.shape == im.shape