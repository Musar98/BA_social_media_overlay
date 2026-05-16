"""
Unit tests for the conditional parametric generator classes.
These tests monkeypatch MobileNet creation and image transformation helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type

import pytest
import torch
import torch.nn as nn

from social_media_overlay.models.parametric import (
    ConditionalParametricConvGenerator,
    ConditionalParametricMobileNetGenerator,
    ConditionalParametricResidualMobileNetGenerator,
)

import social_media_overlay.models.parametric as generator_module


BATCH_SIZE = 2
IN_CHANNELS = 3
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 16
CURVE_STEPS = 8

EXPECTED_PARAM_SHAPES = {
    "exposure": (BATCH_SIZE,),
    "saturation": (BATCH_SIZE,),
    "tone": (BATCH_SIZE, 1, CURVE_STEPS, 1),
    "color": (BATCH_SIZE, 3, CURVE_STEPS, 1),
    "contrast": (BATCH_SIZE,),
    "sharp": (BATCH_SIZE,),
    "blur": (BATCH_SIZE,),
}

EXPECTED_FLAT_PARAM_DIM = (
    1  # exposure
    + 1  # saturation
    + CURVE_STEPS  # tone
    + 3 * CURVE_STEPS  # color
    + 1  # contrast
    + 1  # sharp
    + 1  # blur
)


class TinyBackbone(nn.Module):
    """Small stand-in for timm MobileNet backbones used by the unit tests."""

    out_channels = 8

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


@dataclass(frozen=True)
class ModelCase:
    model_cls: Type[nn.Module]
    kwargs: dict[str, Any]
    alpha_factory: Callable[[int], torch.Tensor]


@pytest.fixture(autouse=True)
def patch_generator_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Patch expensive/external dependencies in the generator module."""

    module = generator_module

    def fake_apply_params(x: torch.Tensor, params: dict[str, torch.Tensor]):
        # Preserve the public contract: apply_params returns intermediate images,
        # and the generator returns the final image at img_list[-1].
        return [x, x.clamp(0.0, 1.0)]

    def fake_flatten_output_params(
        params: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        flat_parts = []
        for value in params.values():
            flat_parts.append(value.reshape(batch_size, -1))
        return torch.cat(flat_parts, dim=1)

    def fake_create_mobilenet_backbone(
        mobilenet_variant: str,
        in_channels: int,
        mobilenet_pretrained: bool,
    ):
        return TinyBackbone(in_channels=in_channels), {
            "input_size": (in_channels, IMAGE_HEIGHT, IMAGE_WIDTH),
            "mean": (0.0,) * in_channels,
            "std": (1.0,) * in_channels,
        }

    def fake_resolve_backbone_stats_and_input_size(
        data_cfg: dict[str, Any],
        in_channels: int,
        mobilenet_pretrained: bool,
        backbone_input_size: int | None,
    ):
        input_size = backbone_input_size or IMAGE_HEIGHT
        return (0.0,) * in_channels, (1.0,) * in_channels, (input_size, input_size)

    def fake_infer_mobilenet_encoder_dim(
        backbone: TinyBackbone,
        in_channels: int,
        backbone_input_hw: tuple[int, int],
        backbone_mean: tuple[float, ...],
        backbone_std: tuple[float, ...],
        mobilenet_pretrained: bool,
    ) -> int:
        return backbone.out_channels

    def fake_normalize_for_mobilenet(
        x: torch.Tensor,
        backbone_input_hw: tuple[int, int],
    ) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x,
            size=backbone_input_hw,
            mode="bilinear",
            align_corners=False,
        )

    monkeypatch.setattr(module, "apply_params", fake_apply_params)
    monkeypatch.setattr(module, "flatten_output_params", fake_flatten_output_params)
    monkeypatch.setattr(module, "create_mobilenet_backbone", fake_create_mobilenet_backbone)
    monkeypatch.setattr(
        module,
        "resolve_backbone_stats_and_input_size",
        fake_resolve_backbone_stats_and_input_size,
    )
    monkeypatch.setattr(
        module,
        "infer_mobilenet_encoder_dim",
        fake_infer_mobilenet_encoder_dim,
    )
    monkeypatch.setattr(module, "normalize_for_mobilenet", fake_normalize_for_mobilenet)


@pytest.fixture()
def image_batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(BATCH_SIZE, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)


MODEL_CASES = [
    ModelCase(
        model_cls=ConditionalParametricConvGenerator,
        kwargs={
            "num_features": 4,
            "fc_dim": 16,
            "num_conv_layers": 2,
            "do_norm": False,
            "in_channels": IN_CHANNELS,
            "cond_dim": 4,
        },
        alpha_factory=lambda batch_size: torch.linspace(0.0, 1.0, batch_size),
    ),
    ModelCase(
        model_cls=ConditionalParametricMobileNetGenerator,
        kwargs={
            "fc_width": 16,
            "fc_depth": 1,
            "in_channels": IN_CHANNELS,
            "mobilenet_pretrained": False,
            "freeze_backbone": True,
            "backbone_input_size": IMAGE_HEIGHT,
            "cond_dim": 4,
        },
        alpha_factory=lambda batch_size: torch.linspace(0.0, 1.0, batch_size),
    ),
    ModelCase(
        model_cls=ConditionalParametricResidualMobileNetGenerator,
        kwargs={
            "fc_dim": 16,
            "fc_depth": 1,
            "in_channels": IN_CHANNELS,
            "mobilenet_pretrained": False,
            "freeze_backbone": True,
            "backbone_input_size": IMAGE_HEIGHT,
            "cond_dim": 4,
        },
        alpha_factory=lambda batch_size: torch.linspace(0.0, 1.0, batch_size),
    ),
]


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_can_be_instantiated(case: ModelCase):
    model = case.model_cls(**case.kwargs)

    assert isinstance(model, nn.Module)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_image_encoder_returns_expected_dimensions(
    image_batch: torch.Tensor,
    case: ModelCase,
):
    model = case.model_cls(**case.kwargs)

    with torch.no_grad():
        img_features = model.get_img_enc(image_batch)

    assert img_features.ndim == 2
    assert img_features.shape[0] == BATCH_SIZE

    if case.model_cls is ConditionalParametricConvGenerator:
        # With num_features=4 and num_conv_layers=2, the fake conv encoder should
        # return a fixed 2D feature tensor. We do not hardcode the second
        # dimension because it depends on build_conv_encoder internals.
        assert img_features.shape[1] == model.dense0.in_features - model.cond_dim
    else:
        assert img_features.shape == (BATCH_SIZE, TinyBackbone.out_channels)


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_predicted_params_return_expected_keys_and_dimensions(
    image_batch: torch.Tensor,
    case: ModelCase,
):
    model = case.model_cls(**case.kwargs)
    alpha = case.alpha_factory(BATCH_SIZE)

    with torch.no_grad():
        img_features = model.get_img_enc(image_batch)
        params = model.get_predicted_params(img_features, alpha)

    assert set(params.keys()) == set(EXPECTED_PARAM_SHAPES.keys())

    for name, expected_shape in EXPECTED_PARAM_SHAPES.items():
        assert params[name].shape == expected_shape
        assert torch.isfinite(params[name]).all()


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_predicted_params_accept_column_alpha_and_return_expected_dimensions(
    image_batch: torch.Tensor,
    case: ModelCase,
):
    model = case.model_cls(**case.kwargs)
    alpha = case.alpha_factory(BATCH_SIZE).unsqueeze(1)

    with torch.no_grad():
        img_features = model.get_img_enc(image_batch)
        params = model.get_predicted_params(img_features, alpha)

    assert set(params.keys()) == set(EXPECTED_PARAM_SHAPES.keys())

    for name, expected_shape in EXPECTED_PARAM_SHAPES.items():
        assert params[name].shape == expected_shape
        assert torch.isfinite(params[name]).all()


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_simple_train_forward(image_batch: torch.Tensor, case: ModelCase):
    model = case.model_cls(**case.kwargs)
    alpha = case.alpha_factory(BATCH_SIZE)

    model.train()
    out, flat_params = model(image_batch, alpha)

    assert model.training is True

    assert out.shape == (BATCH_SIZE, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert flat_params.shape == (BATCH_SIZE, EXPECTED_FLAT_PARAM_DIM)

    assert out.dtype == image_batch.dtype
    assert flat_params.dtype == image_batch.dtype

    assert torch.isfinite(out).all()
    assert torch.isfinite(flat_params).all()

    assert model.img_list is not None
    assert len(model.img_list) == 2
    assert model.img_list[-1].shape == out.shape


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_simple_inference_forward(image_batch: torch.Tensor, case: ModelCase):
    model = case.model_cls(**case.kwargs)
    alpha = case.alpha_factory(BATCH_SIZE).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        out, flat_params = model(image_batch, alpha)

    assert model.training is False

    assert out.shape == (BATCH_SIZE, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert flat_params.shape == (BATCH_SIZE, EXPECTED_FLAT_PARAM_DIM)

    assert out.dtype == image_batch.dtype
    assert flat_params.dtype == image_batch.dtype

    assert torch.isfinite(out).all()
    assert torch.isfinite(flat_params).all()

    assert model.img_list is not None
    assert len(model.img_list) == 2
    assert model.img_list[-1].shape == out.shape


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_forward_supports_different_batch_size(case: ModelCase):
    batch_size = 3
    image = torch.rand(batch_size, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    alpha = case.alpha_factory(batch_size)

    model = case.model_cls(**case.kwargs)

    with torch.no_grad():
        out, flat_params = model(image, alpha)

    assert out.shape == (batch_size, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert flat_params.shape == (batch_size, EXPECTED_FLAT_PARAM_DIM)


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.model_cls.__name__)
def test_generator_forward_uses_only_first_three_channels(case: ModelCase):
    batch_size = 2
    image = torch.rand(batch_size, 4, IMAGE_HEIGHT, IMAGE_WIDTH)
    alpha = case.alpha_factory(batch_size)

    kwargs = dict(case.kwargs)
    kwargs["in_channels"] = 4

    model = case.model_cls(**kwargs)

    with torch.no_grad():
        out, flat_params = model(image, alpha)

    assert out.shape == (batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert flat_params.shape == (batch_size, EXPECTED_FLAT_PARAM_DIM)


def test_residual_generator_keeps_frozen_backbone_in_eval_mode():
    model = ConditionalParametricResidualMobileNetGenerator(
        fc_dim=16,
        fc_depth=1,
        in_channels=IN_CHANNELS,
        mobilenet_pretrained=False,
        freeze_backbone=True,
        backbone_input_size=IMAGE_HEIGHT,
        cond_dim=4,
    )

    model.train()

    assert model.training is True
    assert model.backbone.training is False


def test_mobilenet_generator_keeps_frozen_backbone_parameters_frozen():
    model = ConditionalParametricMobileNetGenerator(
        fc_width=16,
        fc_depth=1,
        in_channels=IN_CHANNELS,
        mobilenet_pretrained=False,
        freeze_backbone=True,
        backbone_input_size=IMAGE_HEIGHT,
        cond_dim=4,
    )

    assert all(parameter.requires_grad is False for parameter in model.backbone.parameters())


def test_residual_generator_keeps_frozen_backbone_parameters_frozen():
    model = ConditionalParametricResidualMobileNetGenerator(
        fc_dim=16,
        fc_depth=1,
        in_channels=IN_CHANNELS,
        mobilenet_pretrained=False,
        freeze_backbone=True,
        backbone_input_size=IMAGE_HEIGHT,
        cond_dim=4,
    )

    assert all(parameter.requires_grad is False for parameter in model.backbone.parameters())