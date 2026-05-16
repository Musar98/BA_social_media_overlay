# test_clip_loss.py

import pytest
import torch

from social_media_overlay.clip_loss import (
    CLIP_MEAN,
    CLIP_STD,
    CLIP_INPUT_SIZE,
    normalize_for_clip,
    compute_clip_semantic_loss,
)


CLIP_MODEL_NAME = "ViT-B/32"


class DummyClipModel(torch.nn.Module):
    """
    Minimal fake CLIP model for unit tests.

    encode_image returns deterministic embeddings based on the input tensor.
    """

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Use simple deterministic features derived from image statistics.
        mean = x.mean(dim=(1, 2, 3))
        std = x.std(dim=(1, 2, 3))

        features = torch.stack(
            [
                mean,
                std,
                mean + std,
                mean - std,
            ],
            dim=-1,
        )

        assert features.shape == (batch_size, 4)
        return features


def test_normalize_for_clip_resizes_to_clip_input_size():
    x = torch.rand(2, 3, 64, 80)

    out = normalize_for_clip(x, clip_input_size=224)

    assert out.shape == (2, 3, 224, 224)


def test_normalize_for_clip_uses_clip_mean_and_std_for_constant_image():
    x = torch.ones(1, 3, 224, 224)

    out = normalize_for_clip(x, clip_input_size=224)

    expected = torch.empty_like(out)
    for channel in range(3):
        expected[:, channel, :, :] = (1.0 - CLIP_MEAN[channel]) / CLIP_STD[channel]

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_normalize_for_clip_preserves_dtype_and_device():
    x = torch.rand(1, 3, 32, 32, dtype=torch.float64)

    out = normalize_for_clip(x, clip_input_size=224)

    assert out.dtype == torch.float64
    assert out.device == x.device


def test_compute_clip_semantic_loss_returns_scalar_tensor():
    clip_model = DummyClipModel()

    original = torch.rand(4, 3, 64, 64)
    transformed = torch.rand(4, 3, 64, 64)

    loss = compute_clip_semantic_loss(
        original,
        transformed,
        clip_model,
        clip_input_size=224,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_compute_clip_semantic_loss_is_near_zero_for_identical_images():
    clip_model = DummyClipModel()

    images = torch.rand(4, 3, 64, 64)

    loss = compute_clip_semantic_loss(
        images,
        images,
        clip_model,
        clip_input_size=224,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1e-5, atol=1e-5)


def test_compute_clip_semantic_loss_matches_manual_cosine_distance():
    clip_model = DummyClipModel()

    original = torch.rand(3, 3, 32, 32)
    transformed = torch.rand(3, 3, 32, 32)

    loss = compute_clip_semantic_loss(
        original,
        transformed,
        clip_model,
        clip_input_size=224,
    )

    x_orig = normalize_for_clip(original, clip_input_size=224)
    x_trans = normalize_for_clip(transformed, clip_input_size=224)

    feat_orig = clip_model.encode_image(x_orig)
    feat_trans = clip_model.encode_image(x_trans)

    feat_orig = torch.nn.functional.normalize(feat_orig, dim=-1)
    feat_trans = torch.nn.functional.normalize(feat_trans, dim=-1)

    per_sample_loss = 1.0 - (feat_orig * feat_trans).sum(dim=-1)
    expected_loss = per_sample_loss.mean()

    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)


def test_compute_clip_semantic_loss_ignores_zero_alpha_when_requested():
    clip_model = DummyClipModel()

    original = torch.rand(3, 3, 32, 32)
    transformed = torch.rand(3, 3, 32, 32)
    alphas = torch.tensor([1.0, 0.0, 1.0])

    loss = compute_clip_semantic_loss(
        original,
        transformed,
        clip_model,
        clip_input_size=224,
        alphas=alphas,
        loss_on_zero_alpha=False,
    )

    x_orig = normalize_for_clip(original, clip_input_size=224)
    x_trans = normalize_for_clip(transformed, clip_input_size=224)

    feat_orig = torch.nn.functional.normalize(clip_model.encode_image(x_orig), dim=-1)
    feat_trans = torch.nn.functional.normalize(clip_model.encode_image(x_trans), dim=-1)

    per_sample_loss = 1.0 - (feat_orig * feat_trans).sum(dim=-1)

    expected_loss = torch.stack(
        [
            per_sample_loss[0],
            per_sample_loss[2],
        ]
    ).mean()

    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)


def test_compute_clip_semantic_loss_all_zero_alpha_returns_zero_when_ignored():
    clip_model = DummyClipModel()

    original = torch.rand(3, 3, 32, 32)
    transformed = torch.rand(3, 3, 32, 32)
    alphas = torch.zeros(3)

    loss = compute_clip_semantic_loss(
        original,
        transformed,
        clip_model,
        clip_input_size=224,
        alphas=alphas,
        loss_on_zero_alpha=False,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1e-6, atol=1e-6)


def test_compute_clip_semantic_loss_scales_by_absolute_alpha():
    clip_model = DummyClipModel()

    original = torch.rand(3, 3, 32, 32)
    transformed = torch.rand(3, 3, 32, 32)
    alphas = torch.tensor([1.0, -2.0, 0.0])

    loss = compute_clip_semantic_loss(
        original,
        transformed,
        clip_model,
        clip_input_size=224,
        alphas=alphas,
        loss_on_zero_alpha=True,
        scale_by_alpha=True,
    )

    x_orig = normalize_for_clip(original, clip_input_size=224)
    x_trans = normalize_for_clip(transformed, clip_input_size=224)

    feat_orig = torch.nn.functional.normalize(clip_model.encode_image(x_orig), dim=-1)
    feat_trans = torch.nn.functional.normalize(clip_model.encode_image(x_trans), dim=-1)

    per_sample_loss = 1.0 - (feat_orig * feat_trans).sum(dim=-1)

    weights = alphas.abs()
    expected_loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)

    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)


def test_compute_clip_semantic_loss_zero_weights_returns_zero_when_scaled_by_alpha():
    clip_model = DummyClipModel()

    original = torch.rand(3, 3, 32, 32)
    transformed = torch.rand(3, 3, 32, 32)
    alphas = torch.zeros(3)

    loss = compute_clip_semantic_loss(
        original,
        transformed,
        clip_model,
        clip_input_size=224,
        alphas=alphas,
        loss_on_zero_alpha=True,
        scale_by_alpha=True,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1e-6, atol=1e-6)


def test_compute_clip_semantic_loss_backpropagates_to_images():
    clip_model = DummyClipModel()

    original = torch.rand(2, 3, 32, 32, requires_grad=True)
    transformed = torch.rand(2, 3, 32, 32, requires_grad=True)

    loss = compute_clip_semantic_loss(
        original,
        transformed,
        clip_model,
        clip_input_size=224,
    )

    loss.backward()

    assert original.grad is not None
    assert transformed.grad is not None
    assert original.grad.shape == original.shape
    assert transformed.grad.shape == transformed.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_normalize_for_clip_works_on_cuda():
    x = torch.rand(1, 3, 32, 32, device="cuda")

    out = normalize_for_clip(x, clip_input_size=224)

    assert out.device.type == "cuda"
    assert out.shape == (1, 3, 224, 224)


@pytest.mark.integration
def test_compute_clip_semantic_loss_with_real_clip_vit_b_32():
    """
    Optional integration test.

    Requires:
        pip install git+https://github.com/openai/CLIP.git

    This test loads the actual OpenAI CLIP model.
    It may download model weights the first time it runs.
    """
    clip = pytest.importorskip("clip")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=device)
    clip_model.eval()

    original = torch.rand(2, 3, 128, 128, device=device)
    transformed = original.clone()

    with torch.no_grad():
        loss = compute_clip_semantic_loss(
            original,
            transformed,
            clip_model,
            clip_input_size=CLIP_INPUT_SIZE,
        )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= -1e-5