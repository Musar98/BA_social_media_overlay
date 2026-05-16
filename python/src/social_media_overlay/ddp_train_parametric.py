"""
Training and validation utilities for the single-condition parametric model.

This file contains the training loop, validation loop, and helper functions used
by the conditional parametric image model. The model receives an image and an
alpha value, edits the image through learned parametric transformations, and is
trained so that the edited image moves toward a target emotion score.

The main training signal comes from a pretrained emotion regressor. For each
input image, the regressor first predicts the original emotion score. A sampled
alpha value is then added to this score to create the target score. The
parametric model edits the image, the regressor predicts the emotion of the
edited image, and the model is optimized so the edited score matches the target.

Optional auxiliary losses are also supported:

1. Identity loss:
   Encourages the model to keep the image unchanged when alpha is zero.
   We always use 0.

2. CLIP semantic preservation loss:
   Encourages the edited image to stay semantically close to the original image.
   We use a fixed value of 0.05

The file also contains deterministic validation utilities. Validation saves and
restores random number generator states, sets a fixed validation seed, averages
metrics across batches, and then averages metrics across distributed DDP ranks.
"""
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from social_media_overlay.ddp import get_base_model, safe_checkpoint
from social_media_overlay.utils import to_tensor, zero_scalar
from social_media_overlay.alphas import sample_alphas
from social_media_overlay.clip_loss import compute_clip_semantic_loss, normalize_for_clip

# Global constants
# Minimum valid emotion score. The Gebhardt regressor outputs normalized scores,
# so targets are clipped to this lower bound when CLIP_TARGET is enabled.
SCORE_MIN = 0.0
# Maximum valid emotion score. Target scores above this value are clipped because
# the regressor target space is expected to stay inside [0, 1].
SCORE_MAX = 1.0



def _predict_scores(
    images: torch.Tensor,
    regressor: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    use_arousal: bool,
) -> torch.Tensor:
    """
    Predict emotion scores for a batch of images.

    The regressor returns at least two emotion dimensions. By default, this
    helper uses the first output dimension. If use_arousal=True, it uses the
    second output dimension instead.

    Args:
        images:
            Image tensor, usually shape [B, 3, H, W].

        regressor:
            Pretrained emotion regressor with a predict method.

        device:
            Device on which the returned tensor should be placed.

        dtype:
            Data type used for the returned tensor.

        use_arousal:
            If True, use the second regressor output. If False, use the first
            output.

    Returns:
        Emotion score tensor of shape [B, 1].
    """
    if use_arousal:
        scores = regressor.predict(images)[:, 1:2]
    else:
        scores = regressor.predict(images)[:, :1]
    return to_tensor(scores, device, dtype)


def _build_target_scores(
    X: torch.Tensor,
    regressor: torch.nn.Module,
    device: torch.device,
    zero_alpha_prob: float,
    use_arousal: bool,
    score_min: float,
    score_max: float,
    clip_target: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build alpha values and target emotion scores for one batch.

    First, the current emotion score of each original image is predicted with the
    frozen regressor. Then an alpha value is sampled and added to the original
    score. This creates the target score that the transformed image should reach.

    Args:
        X:
            Original image batch.

        regressor:
            Pretrained emotion regressor.

        device:
            Device used for sampled alpha values and returned tensors.

        zero_alpha_prob:
            Probability of sampling alpha=0. These samples are useful for
            identity preservation.

        use_arousal:
            If True, use the arousal output of the regressor. If False, use the
            first score output.

        score_min:
            Lower bound for target clipping.

        score_max:
            Upper bound for target clipping.

        clip_target:
            If True, clamp target scores into [score_min, score_max].

    Returns:
        A tuple containing:

        alphas:
            Sampled alpha tensor.

        target_score:
            Target emotion score tensor.
    """
    batch_size = X.size(0)

    with torch.no_grad():
        original_scores = _predict_scores(
            images=X,
            regressor=regressor,
            device=device,
            dtype=X.dtype,
            use_arousal=use_arousal,
        )

        alphas = sample_alphas(batch_size, device, zero_alpha_prob=zero_alpha_prob)
        target_score = original_scores + alphas

        if clip_target:
            target_score = torch.clamp(target_score, min=score_min, max=score_max)

    return alphas, target_score


def _compute_identity_loss(
    transformed_images: torch.Tensor,
    original_images: torch.Tensor,
    alphas: torch.Tensor,
    identity_loss_weight: float,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Compute image-space identity loss for alpha=0 samples.

    The identity loss is only applied to samples where alpha is exactly zero. For
    those samples, the transformed image should stay close to the original image.

    Args:
        transformed_images:
            Images produced by the parametric model.

        original_images:
            Original input images.

        alphas:
            Alpha values used for this batch.

        identity_loss_weight:
            Weight of the identity loss. If this is <= 0, the loss is disabled.

    Returns:
        A tuple containing:

        zero_mask:
            Mask identifying alpha=0 samples, or None when identity loss is
            disabled.

        identity_loss:
            Mean L1 image loss over alpha=0 samples, or a zero scalar if no such
            samples exist.
    """
    device = original_images.device
    dtype = original_images.dtype

    if identity_loss_weight <= 0.0:
        return None, zero_scalar(device, dtype)

    zero_mask = (alphas == 0).view(-1, 1, 1, 1).to(dtype=dtype)

    if zero_mask.sum().item() > 0:
        abs_diff = (transformed_images - original_images).abs()
        denom = (
            zero_mask.sum()
            * original_images.shape[1]
            * original_images.shape[2]
            * original_images.shape[3]
        )
        identity_loss = (abs_diff * zero_mask).sum() / denom
    else:
        identity_loss = zero_scalar(device, dtype)

    return zero_mask, identity_loss


def _compute_zero_alpha_image_l1(
    transformed_images: torch.Tensor,
    original_images: torch.Tensor,
    zero_mask: torch.Tensor | None,
    identity_loss_weight: float,
) -> torch.Tensor:
    """
    Compute the zero-alpha image L1 metric.

    This metric mirrors the identity loss calculation but is used for logging and
    validation metrics. It reports how much the model changes the image when
    alpha=0.

    Args:
        transformed_images:
            Images produced by the parametric model.

        original_images:
            Original input images.

        zero_mask:
            Mask identifying alpha=0 samples.

        identity_loss_weight:
            Identity loss weight. If the identity loss is disabled, this metric
            returns zero as well.

    Returns:
        Mean L1 difference for alpha=0 samples, or a zero scalar.
    """
    device = original_images.device
    dtype = original_images.dtype

    if identity_loss_weight <= 0.0 or zero_mask is None:
        return zero_scalar(device, dtype)

    if zero_mask.sum().item() > 0:
        return ((transformed_images - original_images).abs() * zero_mask).sum() / (
            zero_mask.sum()
            * original_images.shape[1]
            * original_images.shape[2]
            * original_images.shape[3]
        )

    return zero_scalar(device, dtype)


def _compute_clip_loss(
    X: torch.Tensor,
    transformed_images: torch.Tensor,
    clip_model: torch.nn.Module | None,
    clip_semantic_loss_weight: float,
    clip_input_size: int,
    alphas: torch.Tensor,
    clip_loss_on_zero_alpha: bool,
    clip_loss_scale_by_alpha: bool,
) -> torch.Tensor:
    """
    Compute the optional CLIP semantic preservation loss.

    This loss compares the original and transformed images in CLIP feature space.
    It encourages the model to preserve semantic image content while changing the
    emotion score.

    Args:
        X:
            Original image batch.

        transformed_images:
            Images produced by the parametric model.

        clip_model:
            Frozen CLIP model. Required when clip_semantic_loss_weight > 0.

        clip_semantic_loss_weight:
            If <= 0, CLIP loss is disabled and this function returns zero.

        clip_input_size:
            Input resolution expected by the selected CLIP model.

        alphas:
            Alpha values used for this batch.

        clip_loss_on_zero_alpha:
            If True, also apply CLIP loss to alpha=0 samples.

        clip_loss_scale_by_alpha:
            If True, scale CLIP loss according to alpha strength.

    Returns:
        CLIP semantic loss tensor, or a zero scalar if disabled.
    """
    device = X.device
    dtype = X.dtype

    if clip_semantic_loss_weight <= 0.0:
        return zero_scalar(device, dtype)

    if clip_model is None:
        raise ValueError(
            "clip_model must be provided when clip_semantic_loss_weight > 0."
        )

    return compute_clip_semantic_loss(
        original_images=X,
        transformed_images=transformed_images,
        clip_model=clip_model,
        clip_input_size=clip_input_size,
        alphas=alphas,
        loss_on_zero_alpha=clip_loss_on_zero_alpha,
        scale_by_alpha=clip_loss_scale_by_alpha,
    )


def _compute_aux_metrics(
    X: torch.Tensor,
    transformed_images: torch.Tensor,
    transformed_scores: torch.Tensor,
    target_score: torch.Tensor,
    zero_mask: torch.Tensor | None,
    identity_loss_weight: float,
    score_min: float,
    score_max: float,
) -> dict[str, torch.Tensor]:
    """
    Compute auxiliary metrics for logging and validation.

    These metrics are not directly optimized, but they help interpret model
    behavior during training and validation.

    Args:
        X:
            Original image batch.

        transformed_images:
            Images produced by the parametric model.

        transformed_scores:
            Emotion scores predicted for transformed images.

        target_score:
            Target emotion scores.

        zero_mask:
            Mask identifying alpha=0 samples.

        identity_loss_weight:
            Identity loss weight.

        score_min:
            Minimum valid score.

        score_max:
            Maximum valid score.

    Returns:
        Dictionary containing auxiliary metric tensors.
    """
    with torch.no_grad():
        score_abs_error = (transformed_scores - target_score).abs().mean()
        image_l1 = (transformed_images - X).abs().mean()
        zero_alpha_image_l1 = _compute_zero_alpha_image_l1(
            transformed_images=transformed_images,
            original_images=X,
            zero_mask=zero_mask,
            identity_loss_weight=identity_loss_weight,
        )
        clipped_target_fraction = (
            ((target_score <= score_min) | (target_score >= score_max)).to(X.dtype).mean()
        )

    return {
        "score_abs_error": score_abs_error,
        "image_l1": image_l1,
        "zero_alpha_image_l1": zero_alpha_image_l1,
        "clipped_target_fraction": clipped_target_fraction,
    }


def _init_metric_sums() -> dict[str, float]:
    """
    Initialize validation metric accumulators.

    Returns:
        Dictionary where every tracked metric starts at zero.
    """
    return {
        "loss": 0.0,
        "score_loss": 0.0,
        "identity_loss": 0.0,
        "clip_semantic_loss": 0.0,
        "score_abs_error": 0.0,
        "image_l1": 0.0,
        "zero_alpha_image_l1": 0.0,
        "clipped_target_fraction": 0.0,
    }


def _average_metric_sums(
    metric_sums: dict[str, float],
    num_batches: int,
) -> dict[str, float]:
    """
    Convert summed validation metrics into per-batch averages.

    Args:
        metric_sums:
            Dictionary containing accumulated metric values.

        num_batches:
            Number of validation batches.

    Returns:
        Dictionary containing averaged metric values.
    """
    return {
        key: value / max(num_batches, 1)
        for key, value in metric_sums.items()
    }


def _distributed_average_metrics(
    metric_avgs: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    """
    Average validation metrics across all DDP ranks.

    Each rank computes local validation averages. This helper uses all_reduce so
    the returned metrics represent the average across all distributed processes.

    Args:
        metric_avgs:
            Local metric averages from one rank.

        device:
            Device used for the temporary reduction tensor.

    Returns:
        Metric averages across all ranks.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return metric_avgs

    keys = list(metric_avgs.keys())
    values = torch.tensor([metric_avgs[k] for k in keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / dist.get_world_size()

    return {k: values[i].item() for i, k in enumerate(keys)}


def _save_rng_states() -> dict[str, Any]:
    """
    Save random number generator states.

    Validation uses a fixed seed for reproducibility. To avoid disturbing the
    training randomness, this helper stores the current RNG states before the
    validation seed is set.

    Returns:
        Dictionary containing Python, NumPy, PyTorch, and optional CUDA RNG
        states.
    """
    states = {
        "torch": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": None,
    }

    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()

    return states


def _restore_rng_states(states: dict[str, Any]) -> None:
    """
    Restore previously saved random number generator states.

    Args:
        states:
            RNG state dictionary returned by _save_rng_states.
    """
    torch.random.set_rng_state(states["torch"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])

    if torch.cuda.is_available() and states["cuda"] is not None:
        torch.cuda.set_rng_state_all(states["cuda"])


def _set_validation_seed(val_seed: int) -> None:
    """
    Set a deterministic random seed for validation.

    This makes alpha sampling and any other random validation behavior
    reproducible across validation runs.

    Args:
        val_seed:
            Seed value used for PyTorch, NumPy, Python, and CUDA RNGs.
    """
    torch.manual_seed(val_seed)
    np.random.seed(val_seed)
    random.seed(val_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(val_seed)


def compute_loss(
    X: torch.Tensor,
    model: torch.nn.Module,
    regressor: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    zero_alpha_prob: float = 0.1,
    identity_loss_weight: float = 0.1,
    use_arousal: bool = False,
    score_min: float = SCORE_MIN,
    score_max: float = SCORE_MAX,
    clip_target: bool = True,
    clip_model: torch.nn.Module | None = None,
    clip_semantic_loss_weight: float = 0.0,
    clip_input_size: int = 224,
    clip_loss_on_zero_alpha: bool = True,
    clip_loss_scale_by_alpha: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute all loss terms and metrics for one training or validation batch.

    This function builds target emotion scores, transforms the image with the
    parametric model, predicts the emotion score of the transformed image, and
    combines the score loss with optional identity and CLIP losses.

    Args:
        X:
            Original image batch.

        model:
            Parametric generator model.

        regressor:
            Frozen emotion regressor.

        loss_fn:
            Main score loss function, usually MSELoss.

        device:
            Training device.

        zero_alpha_prob:
            Probability of sampling alpha=0.

        identity_loss_weight:
            Weight applied to identity_loss in the final total loss.

        use_arousal:
            If True, optimize the arousal regressor output. If False, optimize
            the first score output.

        score_min:
            Lower valid emotion score.

        score_max:
            Upper valid emotion score.

        clip_target:
            If True, target scores are clipped to [score_min, score_max].

        clip_model:
            Optional frozen CLIP model.

        clip_semantic_loss_weight:
            Weight applied to CLIP semantic preservation loss. If 0, CLIP loss is
            disabled.

        clip_input_size:
            Input size expected by the selected CLIP model.

        clip_loss_on_zero_alpha:
            Whether CLIP loss should also apply to alpha=0 samples.

        clip_loss_scale_by_alpha:
            Whether CLIP loss should be scaled by alpha strength.

    Returns:
        Dictionary containing total loss, component losses, and auxiliary
        metrics. Values are tensors so they can still be used for backpropagation
        where needed.
    """
    alphas, target_score = _build_target_scores(
        X=X,
        regressor=regressor,
        device=device,
        zero_alpha_prob=zero_alpha_prob,
        use_arousal=use_arousal,
        score_min=score_min,
        score_max=score_max,
        clip_target=clip_target,
    )

    transformed_images, _ = model(X, alphas)

    transformed_scores = _predict_scores(
        images=transformed_images,
        regressor=regressor,
        device=device,
        dtype=X.dtype,
        use_arousal=use_arousal,
    )

    score_loss = loss_fn(transformed_scores, target_score)

    zero_mask, identity_loss = _compute_identity_loss(
        transformed_images=transformed_images,
        original_images=X,
        alphas=alphas,
        identity_loss_weight=identity_loss_weight,
    )

    clip_semantic_loss = _compute_clip_loss(
        X=X,
        transformed_images=transformed_images,
        clip_model=clip_model,
        clip_semantic_loss_weight=clip_semantic_loss_weight,
        clip_input_size=clip_input_size,
        alphas=alphas,
        clip_loss_on_zero_alpha=clip_loss_on_zero_alpha,
        clip_loss_scale_by_alpha=clip_loss_scale_by_alpha,
    )

    total_loss = (
        score_loss
        + identity_loss_weight * identity_loss
        + clip_semantic_loss_weight * clip_semantic_loss
    )

    metrics = _compute_aux_metrics(
        X=X,
        transformed_images=transformed_images,
        transformed_scores=transformed_scores,
        target_score=target_score,
        zero_mask=zero_mask,
        identity_loss_weight=identity_loss_weight,
        score_min=score_min,
        score_max=score_max,
    )

    return {
        "loss": total_loss,
        "score_loss": score_loss,
        "identity_loss": identity_loss,
        "clip_semantic_loss": clip_semantic_loss,
        **metrics,
    }


def train(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    regressor: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_state: bool,
    save_path: str,
    save_name: str,
    rank: int,
    loss_history: list[float],
    val_loss_history: list[dict[str, float]],
    epoch: int = 0,
    start_batch: int = 0,
    print_every: int = 100,
    zero_alpha_prob: float = 0.1,
    identity_loss_weight: float = 0.1,
    use_arousal: bool = False,
    score_min: float = SCORE_MIN,
    score_max: float = SCORE_MAX,
    clip_target: bool = True,
    clip_model: torch.nn.Module | None = None,
    clip_semantic_loss_weight: float = 0.0,
    clip_input_size: int = 224,
    clip_loss_on_zero_alpha: bool = True,
    clip_loss_scale_by_alpha: bool = False,
) -> list[float]:
    """
    Run one training epoch.

    The function iterates over the dataloader, optionally skips already completed
    batches when resuming, computes the full loss dictionary, performs
    backpropagation, logs metrics, and optionally writes checkpoints.

    Args:
        dataloader:
            Training DataLoader.

        model:
            DDP-wrapped or plain parametric generator.

        regressor:
            Frozen emotion regressor.

        loss_fn:
            Main score loss function.

        optimizer:
            Optimizer for trainable model parameters.

        device:
            Device used by this process.

        save_state:
            If True, save periodic checkpoints from rank 0.

        save_path:
            Directory where checkpoints are written.

        save_name:
            Prefix used for checkpoint filenames.

        rank:
            Global DDP rank of this process.

        loss_history:
            List of scalar training losses. This is appended to in-place.

        val_loss_history:
            Validation metrics saved together with checkpoints.

        epoch:
            Current epoch index.

        start_batch:
            Batch index to resume from inside the current epoch.

        print_every:
            Print and checkpoint every N batches.

        zero_alpha_prob:
            Probability of sampling alpha=0.

        identity_loss_weight:
            Weight of the identity loss.

        use_arousal:
            If True, train against the arousal output.

        score_min:
            Minimum valid target score.

        score_max:
            Maximum valid target score.

        clip_target:
            If True, clamp target scores to the valid score range.

        clip_model:
            Optional frozen CLIP model.

        clip_semantic_loss_weight:
            Weight for CLIP semantic loss.

        clip_input_size:
            Input size expected by CLIP.

        clip_loss_on_zero_alpha:
            Whether CLIP loss also applies to alpha=0 samples.

        clip_loss_scale_by_alpha:
            Whether CLIP loss is scaled by alpha strength.

    Returns:
        Updated loss_history list.
    """
    size = len(dataloader.dataset)
    model.train()

    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    base_model = get_base_model(model)

    for batch, (X, _) in enumerate(dataloader):
        if batch < start_batch:
            continue

        X = X.to(device, non_blocking=True)
        batch_size = X.size(0)

        optimizer.zero_grad(set_to_none=True)

        losses = compute_loss(
            X=X,
            model=model,
            regressor=regressor,
            loss_fn=loss_fn,
            device=device,
            zero_alpha_prob=zero_alpha_prob,
            identity_loss_weight=identity_loss_weight,
            use_arousal=use_arousal,
            score_min=score_min,
            score_max=score_max,
            clip_target=clip_target,
            clip_model=clip_model,
            clip_semantic_loss_weight=clip_semantic_loss_weight,
            clip_input_size=clip_input_size,
            clip_loss_on_zero_alpha=clip_loss_on_zero_alpha,
            clip_loss_scale_by_alpha=clip_loss_scale_by_alpha,
        )

        losses["loss"].backward()
        optimizer.step()

        loss_value = losses["loss"].item()
        loss_history.append(loss_value)

        if rank == 0 and batch % print_every == 0:
            current = (batch + 1) * batch_size * dist.get_world_size()
            print(
                f"epoch {epoch} | batch {batch} | loss {loss_value:.6f} "
                f"| score_loss {losses['score_loss'].item():.6f} "
                f"| id_loss {losses['identity_loss'].item():.8f} "
                f"| clip_sem_loss {losses['clip_semantic_loss'].item():.6f} "
                f"| score_mae {losses['score_abs_error'].item():.6f} "
                f"| img_l1 {losses['image_l1'].item():.6f} "
                f"| progress [{current:>5d}/{size:>5d}]"
            )

            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"allocated: {mem_alloc:.3f} GB")
                print(f"reserved : {mem_reserved:.3f} GB")

            if save_state:
                safe_checkpoint(
                    save_path,
                    save_name,
                    epoch,
                    batch,
                    base_model.state_dict(),
                    optimizer.state_dict(),
                    loss_history=loss_history,
                    val_loss_history=val_loss_history,
                    name_suffix=f"-{epoch}-{batch}.pt",
                )

    return loss_history


def val(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    regressor: torch.nn.Module,
    device: torch.device,
    loss_fn: torch.nn.Module,
    zero_alpha_prob: float = 0.2,
    identity_loss_weight: float = 0.1,
    use_arousal: bool = False,
    score_min: float = SCORE_MIN,
    score_max: float = SCORE_MAX,
    clip_target: bool = True,
    clip_model: torch.nn.Module | None = None,
    clip_semantic_loss_weight: float = 0.0,
    clip_input_size: int = 224,
    clip_loss_on_zero_alpha: bool = True,
    clip_loss_scale_by_alpha: bool = False,
    val_seed: int = 1234,
) -> dict[str, float]:
    """
    Run deterministic validation.

    Validation temporarily sets a fixed random seed so alpha sampling is
    reproducible. After validation, the original RNG states are restored so the
    training loop continues as if validation had not changed the random state.

    Args:
        dataloader:
            Validation DataLoader.

        model:
            DDP-wrapped or plain parametric generator.

        regressor:
            Frozen emotion regressor.

        device:
            Device used by this process.

        loss_fn:
            Main score loss function.

        zero_alpha_prob:
            Probability of sampling alpha=0 during validation.

        identity_loss_weight:
            Weight of the identity loss.

        use_arousal:
            If True, validate against the arousal output.

        score_min:
            Minimum valid target score.

        score_max:
            Maximum valid target score.

        clip_target:
            If True, clamp target scores to the valid score range.

        clip_model:
            Optional frozen CLIP model.

        clip_semantic_loss_weight:
            Weight for CLIP semantic preservation loss.

        clip_input_size:
            Input size expected by CLIP.

        clip_loss_on_zero_alpha:
            Whether CLIP loss also applies to alpha=0 samples.

        clip_loss_scale_by_alpha:
            Whether CLIP loss is scaled by alpha strength.

        val_seed:
            Fixed seed used only during validation.

    Returns:
        Dictionary with averaged validation metrics:

        loss:
            Total validation loss.

        score_loss:
            Emotion score loss.

        identity_loss:
            Alpha-zero image preservation loss.

        clip_semantic_loss:
            CLIP semantic preservation loss.

        score_abs_error:
            Mean absolute error between transformed and target scores.

        image_l1:
            Mean L1 difference between transformed and original images.

        zero_alpha_image_l1:
            Mean L1 difference for alpha=0 samples.

        clipped_target_fraction:
            Fraction of targets clipped to score_min or score_max.
    """
    model.eval()

    rng_states = _save_rng_states()
    _set_validation_seed(val_seed)

    metric_sums = _init_metric_sums()
    num_batches = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device, non_blocking=True)

            losses = compute_loss(
                X=X,
                model=model,
                regressor=regressor,
                loss_fn=loss_fn,
                device=device,
                zero_alpha_prob=zero_alpha_prob,
                identity_loss_weight=identity_loss_weight,
                use_arousal=use_arousal,
                score_min=score_min,
                score_max=score_max,
                clip_target=clip_target,
                clip_model=clip_model,
                clip_semantic_loss_weight=clip_semantic_loss_weight,
                clip_input_size=clip_input_size,
                clip_loss_on_zero_alpha=clip_loss_on_zero_alpha,
                clip_loss_scale_by_alpha=clip_loss_scale_by_alpha,
            )

            for key in metric_sums:
                metric_sums[key] += losses[key].item()

            num_batches += 1

    _restore_rng_states(rng_states)

    metric_avgs = _average_metric_sums(metric_sums, num_batches)
    metric_avgs = _distributed_average_metrics(metric_avgs, device)

    return metric_avgs