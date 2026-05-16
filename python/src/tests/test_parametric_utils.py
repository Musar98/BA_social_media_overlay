"""Tests for social_media_overlay.models.parametric_utils.

These tests intentionally mock timm-facing calls where possible so the suite is
fast and does not require network access or pretrained weight downloads.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import social_media_overlay.models.parametric_utils as pu


# ---------------------------------------------------------------------------
# flatten_output_params / unflatten_output_params
# ---------------------------------------------------------------------------


def _full_params(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "exposure": torch.arange(batch_size, dtype=torch.float32),
        "saturation": torch.arange(batch_size, dtype=torch.float32) + 10,
        "tone": torch.arange(batch_size * 1 * pu.CURVE_STEPS * 1, dtype=torch.float32).reshape(
            batch_size, 1, pu.CURVE_STEPS, 1
        ),
        "color": torch.arange(batch_size * pu.COLOR_CHANNELS * pu.CURVE_STEPS * 1, dtype=torch.float32).reshape(
            batch_size, pu.COLOR_CHANNELS, pu.CURVE_STEPS, 1
        ),
        "contrast": torch.arange(batch_size, dtype=torch.float32) + 20,
        "sharp": torch.arange(batch_size, dtype=torch.float32) + 30,
        "blur": torch.arange(batch_size, dtype=torch.float32) + 40,
    }


def test_flatten_output_params_uses_fixed_param_order_not_mapping_order() -> None:
    batch_size = 2
    params = _full_params(batch_size)

    reversed_params = OrderedDict((key, params[key]) for key in reversed(pu.PARAMS))
    flat = pu.flatten_output_params(reversed_params, batch_size=batch_size)

    assert flat.shape == (batch_size, 37)
    assert torch.equal(flat[:, 0], params["exposure"])
    assert torch.equal(flat[:, 1], params["saturation"])
    assert torch.equal(flat[:, 2:10], params["tone"].reshape(batch_size, -1))
    assert torch.equal(flat[:, 10:34], params["color"].reshape(batch_size, -1))
    assert torch.equal(flat[:, 34], params["contrast"])
    assert torch.equal(flat[:, 35], params["sharp"])
    assert torch.equal(flat[:, 36], params["blur"])


def test_flatten_output_params_supports_subset_and_ignores_unknown_keys() -> None:
    batch_size = 3
    params = {
        "unknown": torch.ones(batch_size, 99),
        "blur": torch.arange(batch_size, dtype=torch.float32),
        "tone": torch.ones(batch_size, 1, pu.CURVE_STEPS, 1),
    }

    flat = pu.flatten_output_params(params, batch_size=batch_size)

    # Only tone and blur are known; fixed PARAMS order places tone before blur.
    assert flat.shape == (batch_size, pu.CURVE_STEPS + 1)
    assert torch.equal(flat[:, : pu.CURVE_STEPS], params["tone"].reshape(batch_size, -1))
    assert torch.equal(flat[:, -1], params["blur"])


def test_flatten_output_params_returns_empty_tensor_when_no_known_params() -> None:
    flat = pu.flatten_output_params({}, batch_size=4)

    assert flat.shape == (4, 0)
    assert flat.dtype == torch.float32


def test_unflatten_output_params_reconstructs_all_expected_shapes_and_values() -> None:
    batch_size = 2
    flat = torch.arange(batch_size * 37, dtype=torch.float32).reshape(batch_size, 37)

    params = pu.unflatten_output_params(flat)

    assert list(params.keys()) == pu.PARAMS
    assert torch.equal(params["exposure"], flat[:, 0])
    assert torch.equal(params["saturation"], flat[:, 1])
    assert params["tone"].shape == (batch_size, 1, pu.CURVE_STEPS, 1)
    assert torch.equal(params["tone"].reshape(batch_size, -1), flat[:, 2:10])
    assert params["color"].shape == (batch_size, pu.COLOR_CHANNELS, pu.CURVE_STEPS, 1)
    assert torch.equal(params["color"].reshape(batch_size, -1), flat[:, 10:34])
    assert torch.equal(params["contrast"], flat[:, 34])
    assert torch.equal(params["sharp"], flat[:, 35])
    assert torch.equal(params["blur"], flat[:, 36])


def test_flatten_then_unflatten_roundtrip_for_full_parameter_set() -> None:
    batch_size = 2
    original = _full_params(batch_size)

    reconstructed = pu.unflatten_output_params(
        pu.flatten_output_params(original, batch_size=batch_size)
    )

    for key in pu.PARAMS:
        assert torch.equal(reconstructed[key], original[key])


@pytest.mark.parametrize(
    "bad_flat",
    [
        torch.zeros(37),
        torch.zeros(1, 37, 1),
    ],
)
def test_unflatten_output_params_rejects_non_2d_tensors(bad_flat: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="must be a 2D tensor"):
        pu.unflatten_output_params(bad_flat)


@pytest.mark.parametrize("num_params", [0, 1, 36, 38])
def test_unflatten_output_params_rejects_wrong_number_of_params(num_params: int) -> None:
    with pytest.raises(ValueError, match="exactly 37"):
        pu.unflatten_output_params(torch.zeros(2, num_params))


# ---------------------------------------------------------------------------
# convolutional encoder helpers
# ---------------------------------------------------------------------------


def test_build_conv_block_with_norm_has_expected_layers_and_forward_shape() -> None:
    block = pu.build_conv_block(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        do_norm=True,
    )

    assert isinstance(block, nn.Sequential)
    assert isinstance(block[0], nn.Conv2d)
    assert isinstance(block[1], nn.LeakyReLU)
    assert isinstance(block[2], nn.InstanceNorm2d)
    assert block[0].in_channels == 3
    assert block[0].out_channels == 8
    assert block[0].kernel_size == (3, 3)
    assert block[0].stride == (2, 2)
    assert block[0].padding == (1, 1)

    y = block(torch.randn(2, 3, 16, 16))
    assert y.shape == (2, 8, 8, 8)


def test_build_conv_block_without_norm_uses_identity() -> None:
    block = pu.build_conv_block(3, 8, do_norm=False)

    assert isinstance(block[2], nn.Identity)


def test_build_conv_encoder_constructs_expected_channel_progression() -> None:
    layers, encoder_dim = pu.build_conv_encoder(
        in_channels=3,
        num_features=4,
        num_conv_layers=3,
        do_norm=False,
    )

    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 3
    assert encoder_dim == 16

    conv0, conv1, conv2 = (layer[0] for layer in layers)
    assert conv0.in_channels == 3
    assert conv0.out_channels == 4
    assert conv0.kernel_size == (7, 7)
    assert conv0.padding == (3, 3)

    assert conv1.in_channels == 4
    assert conv1.out_channels == 8
    assert conv1.kernel_size == (3, 3)
    assert conv1.padding == (1, 1)

    assert conv2.in_channels == 8
    assert conv2.out_channels == 16

    x = torch.randn(2, 3, 64, 64)
    for layer in layers:
        x = layer(x)
    assert x.shape == (2, 16, 8, 8)


def test_build_conv_encoder_with_zero_layers_returns_empty_list_and_base_dim() -> None:
    layers, encoder_dim = pu.build_conv_encoder(3, num_features=4, num_conv_layers=0, do_norm=True)

    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 0
    assert encoder_dim == 4


# ---------------------------------------------------------------------------
# MobileNet helpers
# ---------------------------------------------------------------------------


class TinyBackbone(nn.Module):
    def __init__(self, out_channels: int = 13) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, out_channels, kernel_size=1)
        self.conv_head = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm_head = nn.BatchNorm2d(out_channels)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


def test_create_mobilenet_backbone_calls_timm_with_feature_extractor_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    fake_backbone = nn.Identity()
    fake_cfg = {"input_size": (3, 224, 224), "mean": (0.5, 0.5, 0.5), "std": (0.2, 0.2, 0.2)}

    def fake_create_model(*args: object, **kwargs: object) -> nn.Module:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return fake_backbone

    def fake_resolve_model_data_config(model: nn.Module) -> dict[str, object]:
        calls["resolved_model"] = model
        return fake_cfg

    monkeypatch.setattr(pu.timm, "create_model", fake_create_model)
    monkeypatch.setattr(pu, "resolve_model_data_config", fake_resolve_model_data_config)

    backbone, cfg = pu.create_mobilenet_backbone(
        mobilenet_variant="mobilenet_test_variant",
        in_channels=4,
        mobilenet_pretrained=True,
    )

    assert backbone is fake_backbone
    assert cfg is fake_cfg
    assert calls["args"] == ("mobilenet_test_variant",)
    assert calls["kwargs"] == {
        "pretrained": True,
        "in_chans": 4,
        "num_classes": 0,
        "global_pool": "",
    }
    assert calls["resolved_model"] is fake_backbone


def test_resolve_backbone_stats_and_input_size_uses_pretrained_config_values() -> None:
    data_cfg = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": (3, 192, 256),
    }

    mean, std, input_hw = pu.resolve_backbone_stats_and_input_size(
        data_cfg=data_cfg,
        in_channels=3,
        mobilenet_pretrained=True,
    )

    assert mean == data_cfg["mean"]
    assert std == data_cfg["std"]
    assert input_hw == (192, 256)


def test_resolve_backbone_stats_and_input_size_uses_identity_stats_when_not_pretrained() -> None:
    data_cfg = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "input_size": (3, 192, 256),
    }

    mean, std, input_hw = pu.resolve_backbone_stats_and_input_size(
        data_cfg=data_cfg,
        in_channels=4,
        mobilenet_pretrained=False,
        backbone_input_size=128,
    )

    assert mean == (0.0, 0.0, 0.0, 0.0)
    assert std == (1.0, 1.0, 1.0, 1.0)
    assert input_hw == (128, 128)


def test_freeze_mobilenet_backbone_when_requested_freezes_all_params_and_evals() -> None:
    backbone = TinyBackbone(out_channels=5)
    backbone.train()

    pu.freeze_mobilenet_backbone(backbone, freeze_backbone=True)

    assert backbone.training is False
    assert all(not p.requires_grad for p in backbone.parameters())


def test_freeze_mobilenet_backbone_without_full_freeze_only_freezes_known_heads() -> None:
    backbone = TinyBackbone(out_channels=5)

    pu.freeze_mobilenet_backbone(backbone, freeze_backbone=False)

    assert backbone.stem.weight.requires_grad is True
    assert backbone.stem.bias.requires_grad is True
    assert backbone.conv_head.weight.requires_grad is False
    # conv_head bias is intentionally not changed by the implementation.
    assert backbone.conv_head.bias.requires_grad is True
    assert backbone.norm_head.weight.requires_grad is False
    assert backbone.norm_head.bias.requires_grad is False


def test_freeze_mobilenet_backbone_handles_missing_optional_heads() -> None:
    backbone = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1))

    pu.freeze_mobilenet_backbone(backbone, freeze_backbone=False)

    assert all(p.requires_grad for p in backbone.parameters())


def test_infer_mobilenet_encoder_dim_returns_feature_channels_and_restores_train_state() -> None:
    backbone = TinyBackbone(out_channels=11)
    backbone.train()

    dim = pu.infer_mobilenet_encoder_dim(
        backbone=backbone,
        in_channels=3,
        backbone_input_hw=(8, 10),
        backbone_mean=(0.5, 0.5, 0.5),
        backbone_std=(0.25, 0.25, 0.25),
        mobilenet_pretrained=True,
    )

    assert dim == 11
    assert backbone.training is True


def test_infer_mobilenet_encoder_dim_restores_eval_state() -> None:
    backbone = TinyBackbone(out_channels=7)
    backbone.eval()

    dim = pu.infer_mobilenet_encoder_dim(
        backbone=backbone,
        in_channels=3,
        backbone_input_hw=(8, 8),
        backbone_mean=(0.0, 0.0, 0.0),
        backbone_std=(1.0, 1.0, 1.0),
        mobilenet_pretrained=False,
    )

    assert dim == 7
    assert backbone.training is False


def test_infer_mobilenet_encoder_dim_applies_normalization_only_when_pretrained() -> None:
    class RecordingBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy_param = nn.Parameter(torch.ones(()))
            self.seen: torch.Tensor | None = None

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            self.seen = x.detach().clone()
            return torch.zeros(x.shape[0], 6, 2, 2)

    backbone = RecordingBackbone()

    pu.infer_mobilenet_encoder_dim(
        backbone=backbone,
        in_channels=3,
        backbone_input_hw=(4, 4),
        backbone_mean=(0.5, 1.0, 2.0),
        backbone_std=(0.5, 2.0, 4.0),
        mobilenet_pretrained=True,
    )

    assert backbone.seen is not None
    expected = torch.tensor([-1.0, -0.5, -0.5]).view(1, 3, 1, 1).expand(1, 3, 4, 4)
    assert torch.allclose(backbone.seen, expected)

    pu.infer_mobilenet_encoder_dim(
        backbone=backbone,
        in_channels=3,
        backbone_input_hw=(4, 4),
        backbone_mean=(0.5, 1.0, 2.0),
        backbone_std=(0.5, 2.0, 4.0),
        mobilenet_pretrained=False,
    )
    assert backbone.seen is not None
    assert torch.allclose(backbone.seen, torch.zeros(1, 3, 4, 4))


# ---------------------------------------------------------------------------
# conditioning encoder and preprocessing
# ---------------------------------------------------------------------------


def test_build_cond_encoder_has_expected_layers_and_output_shape() -> None:
    encoder = pu.build_cond_encoder(cond_dim=16)

    assert isinstance(encoder, nn.Sequential)
    assert isinstance(encoder[0], nn.Linear)
    assert encoder[0].in_features == 1
    assert encoder[0].out_features == 16
    assert isinstance(encoder[1], nn.LeakyReLU)
    assert isinstance(encoder[2], nn.Linear)
    assert encoder[2].in_features == 16
    assert encoder[2].out_features == 16
    assert isinstance(encoder[3], nn.LeakyReLU)

    y = encoder(torch.randn(5, 1))
    assert y.shape == (5, 16)


def test_normalize_for_mobilenet_matches_bicubic_interpolate_settings() -> None:
    x = torch.arange(1 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5)

    actual = pu.normalize_for_mobilenet(x, dim=(8, 6))
    expected = F.interpolate(
        x,
        size=(8, 6),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )

    assert actual.shape == (1, 3, 8, 6)
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# parameter mapping helpers
# ---------------------------------------------------------------------------


def test_piecewise_tanh_mapping_preserves_identity_at_zero_and_bounds_direction() -> None:
    u = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])

    mapped = pu.piecewise_tanh_mapping(u, min_val=-2.0, max_val=3.0, identity=0.5, scale=1.0)

    assert torch.isclose(mapped[2], torch.tensor(0.5))
    assert torch.all(mapped[:2] < 0.5)
    assert torch.all(mapped[3:] > 0.5)
    assert torch.all(mapped >= -2.0)
    assert torch.all(mapped <= 3.0)
    assert torch.isclose(mapped[0], torch.tensor(-2.0), atol=1e-5)
    assert torch.isclose(mapped[-1], torch.tensor(3.0), atol=1e-5)


def test_piecewise_tanh_mapping_matches_formula_for_positive_and_negative_values() -> None:
    u = torch.tensor([-0.5, 0.5])
    min_val = -3.0
    max_val = 5.0
    identity = 1.0
    scale = 2.0

    mapped = pu.piecewise_tanh_mapping(u, min_val=min_val, max_val=max_val, identity=identity, scale=scale)

    expected_neg = identity + (identity - min_val) * torch.tanh(u[0] / scale)
    expected_pos = identity + (max_val - identity) * torch.tanh(u[1] / scale)
    assert torch.allclose(mapped, torch.stack([expected_neg, expected_pos]))


def test_shifted_sigmoid_mapping_preserves_identity_at_zero_and_stays_in_bounds() -> None:
    u = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])

    mapped = pu.shifted_sigmoid_mapping(u, min_val=-2.0, max_val=5.0, identity=1.0)

    assert torch.isclose(mapped[2], torch.tensor(1.0), atol=1e-6)
    assert torch.all(mapped >= -2.0)
    assert torch.all(mapped <= 5.0)
    assert torch.all(mapped[:-1] <= mapped[1:])
    assert torch.isclose(mapped[0], torch.tensor(-2.0), atol=1e-5)
    assert torch.isclose(mapped[-1], torch.tensor(5.0), atol=1e-5)


def test_shifted_sigmoid_mapping_preserves_dtype_and_device() -> None:
    u = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)

    mapped = pu.shifted_sigmoid_mapping(u, min_val=0.0, max_val=2.0, identity=0.25)

    assert mapped.dtype == torch.float64
    assert mapped.device == u.device


def test_validate_mapping_bounds_accepts_identity_strictly_inside_range() -> None:
    pu.validate_mapping_bounds(min_val=-1.0, max_val=1.0, identity=0.0)


@pytest.mark.parametrize(
    ("min_val", "max_val", "identity"),
    [
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 0.5),
        (0.0, 1.0, -0.1),
        (0.0, 1.0, 1.1),
    ],
)
def test_validate_mapping_bounds_rejects_identity_outside_or_on_bounds(
    min_val: float,
    max_val: float,
    identity: float,
) -> None:
    with pytest.raises(ValueError, match="Need min_val < identity < max_val"):
        pu.validate_mapping_bounds(min_val=min_val, max_val=max_val, identity=identity)
