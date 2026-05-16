import torch

# OpenAI CLIP image normalization stats.
# Correct for models loaded via OpenAI's `clip.load(...)`, e.g.
# RN50, RN101, RN50x4, RN50x16, RN50x64,
# ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px.
# Source: https://github.com/openai/CLIP/issues/20
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_INPUT_SIZE = 224


def normalize_for_clip(
    x: torch.Tensor, clip_input_size: int = CLIP_INPUT_SIZE
) -> torch.Tensor:
    """
    Prepare already preprocessed image tensor for CLIP image encoder.
    In the future use torchvision.transforms.Resize directly

    Args:
        x: already preprocessed (resized and range transformed) image tensor
        clip_input_size: input dimensions of the clip model

    Returns:
        CLIP preprocessed tensor
    """
    x = torch.nn.functional.interpolate(
        x,
        size=(clip_input_size, clip_input_size),
        mode="bicubic",  # https://github.com/openai/CLIP/blob/main/clip/clip.py directly specified by OpenAI CLIP
        align_corners=False,  # standard image-resizing; pixels treated like are rather than fixed corner points
        antialias=True,  # Torchvision documents that for PIL images, antialiasing is always applied for bilinear and bicubic modes
    )

    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

    return (x - mean) / std


def compute_clip_semantic_loss(
    original_images: torch.Tensor,
    transformed_images: torch.Tensor,
    clip_model: torch.nn.Module,
    clip_input_size: int = CLIP_INPUT_SIZE,
    alphas: torch.tensor = None,
    loss_on_zero_alpha: bool = True,
    scale_by_alpha: bool = False,
) -> torch.Tensor:
    """
    Compute a CLIP-based semantic consistency loss between two image batches.

    The loss compares CLIP image embeddings of the original and transformed
    images using cosine similarity:

        loss = 1 - cosine_similarity(orig_features, trans_features)

    Args:
        original_images: Batch of original images.
        transformed_images: Batch of transformed images.
        clip_model: CLIP model used to encode images.
        clip_input_size: Input resolution expected by CLIP.
        alphas: Optional per-sample alpha values for masking or weighting.
        loss_on_zero_alpha: If False, samples with alpha = 0 are ignored.
        scale_by_alpha: If True, weight each sample loss by |alpha|.

    Returns:
        Scalar semantic loss.
    """

    # Resize and normalize both image batches for CLIP
    x_orig = normalize_for_clip(original_images, clip_input_size=clip_input_size)
    x_trans = normalize_for_clip(transformed_images, clip_input_size=clip_input_size)

    # Extract CLIP image embeddings
    feat_orig = clip_model.encode_image(x_orig)
    feat_trans = clip_model.encode_image(x_trans)

    # Normalize features so dot product becomes cosine similarity
    feat_orig = torch.nn.functional.normalize(feat_orig, dim=-1)
    feat_trans = torch.nn.functional.normalize(feat_trans, dim=-1)

    # Per-sample cosine distance: lower means more semantically similar
    per_sample_loss = 1.0 - (feat_orig * feat_trans).sum(dim=-1)

    if alphas is not None:
        # Use absolute alpha values for masking or weighting
        alpha_abs = alphas.view(-1).abs()

        # Ignore samples where alpha is zero
        if not loss_on_zero_alpha:
            nonzero_mask = (alpha_abs > 0).to(per_sample_loss.dtype)
            denom = nonzero_mask.sum().clamp_min(1.0)
            return (per_sample_loss * nonzero_mask).sum() / denom

        # Weight loss by alpha magnitude
        if scale_by_alpha:
            weights = alpha_abs
            denom = weights.sum().clamp_min(1e-8)
            return (per_sample_loss * weights).sum() / denom

    # Default: average loss over the full batch
    return per_sample_loss.mean()
