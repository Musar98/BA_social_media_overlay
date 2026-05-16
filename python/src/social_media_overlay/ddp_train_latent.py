from typing import MutableSequence

import torch
from social_media_overlay.ddp import get_base_model, safe_checkpoint
import torch.distributed as dist
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    optimizer: Optimizer,
    device: torch.device,
    save_state: bool,
    save_path: str,
    save_name: str,
    rank: int,
    loss_history: MutableSequence[float],
    epoch: int = 0,
    start_batch: int = 0,
    print_every: int = 100,
) -> MutableSequence[float]:
    """
    Train the model for one epoch over the provided dataloader.

    The training objective uses the model's original valence score as a fixed
    reference, samples an alpha shift for each input item, and trains the model
    to predict the original score shifted by that alpha value.

    This function also supports distributed training. Rank 0 handles progress
    logging and checkpointing, while all ranks participate in training.

    Args:
        dataloader: DataLoader that yields batches in the form ``(X, _)``.
            The second item is ignored by this training loop.
        model: Model to train. This may be wrapped by DistributedDataParallel or
            a similar wrapper.
        loss_fn: Loss function used to compare transformed scores with target
            scores.
        optimizer: Optimizer used to update the model parameters.
        device: Device where input tensors and sampled alpha values should be
            placed.
        save_state: Whether to write checkpoints during training.
        save_path: Path or directory passed to ``safe_checkpoint``.
        save_name: Base checkpoint name passed to ``safe_checkpoint``.
        rank: Distributed process rank. Only rank 0 logs and saves checkpoints.
        loss_history: Mutable sequence updated in-place with scalar loss values.
        epoch: Current epoch index.
        start_batch: Batch index to resume from. Earlier batches are skipped.
        print_every: Logging and checkpointing interval, measured in batches.

    Returns:
        The same ``loss_history`` object, updated with loss values from this
        training run.
    """
    size = len(dataloader.dataset)
    model.train()

        # Distributed samplers need the epoch value so each epoch gets a different,
    # but still deterministic, sample order. The hasattr check keeps this working
    # for ordinary dataloaders whose samplers do not implement set_epoch.
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    for batch, (X, _) in enumerate(dataloader):
        # When resuming from a checkpoint, skip batches that were already
        # completed. This keeps the original batch numbers intact for logging and
        # checkpoint filenames.
        if batch < start_batch:
            continue

        X = X.to(device, non_blocking=True)
        batch_size = X.size(0)

        optimizer.zero_grad(set_to_none=True)

         # The training model may be wrapped
        base_model = get_base_model(model)

        # These values define the fixed target for the current batch. They are
        # computed without gradient tracking because the reference valence score,
        # sampled alpha values, and target score should not themselves be trained.
        with torch.no_grad():
            original_scores = base_model.valence_score(X)
            alphas = base_model.sample_alpha(batch_size, device=device)
            target_score = original_scores + alphas

        transformed_scores = model(X, alphas)
        loss = loss_fn(transformed_scores, target_score)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)

        if rank == 0 and batch % print_every == 0:
            current = (batch + 1) * batch_size * dist.get_world_size()
            print(
                f"epoch {epoch} | batch {batch} | loss {loss_value:.6f} "
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
                    val_loss_history = None,
                    name_suffix = f"-{epoch}-{batch}.pt"
                )

    return loss_history
