import torch
import torch.nn.functional as F
from social_media_overlay.ddp import get_base_model, safe_checkpoint
import torch.distributed as dist
from social_media_overlay.utils import to_tensor


def sample_alphas(batch_size, device, zero_alpha_prob=0.2, low=-0.5, high=0.5):
    """
    Sample 2D residual conditioning values:
    alpha[:, 0] = valence
    alpha[:, 1] = arousal

    With probability zero_alpha_prob, both are set to zero.
    """
    alphas = torch.empty(batch_size, 2, device=device).uniform_(low, high)

    # sample one mask per sample, not per dimension
    zero_mask = torch.rand(batch_size, 1, device=device) < zero_alpha_prob
    alphas[zero_mask.expand_as(alphas)] = 0.0

    return alphas


def compute_loss(
    X,
    model,
    regressor,
    loss_fn,
    device,
    zero_alpha_prob=0.1,
    identity_loss_weight=0.1,
):
    batch_size = X.size(0)

    with torch.no_grad():
        # assume regressor predicts at least 2 outputs:
        # [:, 0] = valence, [:, 1] = arousal
        original_scores = to_tensor(regressor.predict(X)[:, :2], device, X.dtype)  # [B, 2]

        alphas = sample_alphas(batch_size, device, zero_alpha_prob=zero_alpha_prob)  # [B, 2]
        target_score = original_scores + alphas  # [B, 2]

    transformed_images, _ = model(X, alphas)

    transformed_scores = to_tensor(
        regressor.predict(transformed_images)[:, :2], device, X.dtype
    )  # [B, 2]

    score_loss = loss_fn(transformed_scores, target_score)

    # identity loss only when BOTH valence and arousal are zero
    zero_mask = (alphas == 0).all(dim=1, keepdim=True).view(-1, 1, 1, 1).to(dtype=X.dtype)

    if zero_mask.sum().item() > 0:
        abs_diff = (transformed_images - X).abs()
        denom = zero_mask.sum() * X.shape[1] * X.shape[2] * X.shape[3]
        identity_loss = (abs_diff * zero_mask).sum() / denom
    else:
        identity_loss = torch.tensor(0.0, device=device, dtype=X.dtype)

    total_loss = score_loss + identity_loss_weight * identity_loss

    return {
        "loss": total_loss,
        "score_loss": score_loss,
        "identity_loss": identity_loss,
    }


def train(
    dataloader,
    model,
    regressor,
    loss_fn,
    optimizer,
    device,
    save_state,
    save_path,
    save_name,
    rank,
    loss_history,
    val_loss_history,
    epoch=0,
    start_batch=0,
    print_every=100,
    zero_alpha_prob=0.1,
    identity_loss_weight=0.1,
):
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
    dataloader,
    model,
    regressor,
    device,
    loss_fn,
    zero_alpha_prob=0.2,
    identity_loss_weight=0.1,
):
    model.eval()

    total_loss = 0.0
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
            )

            total_loss += losses["loss"].item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()

    return avg_loss