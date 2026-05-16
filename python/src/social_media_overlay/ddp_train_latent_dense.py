from typing import MutableSequence

import torch
from social_media_overlay.ddp import get_base_model, safe_checkpoint
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module

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

    This function supports distributed training. Only rank 0 prints progress
    information and writes checkpoints. If the dataloader uses a distributed
    sampler, the sampler epoch is set so that shuffling is deterministic but
    different across epochs.

    Args:
        dataloader: DataLoader that yields batches in the form ``(X, _)``.
            The second returned item is ignored by this training loop.
        model: Model to train. This may be a wrapped distributed model.
        loss_fn: Loss function used to compare transformed scores against the
            target scores.
        optimizer: Optimizer used to update model parameters.
        device: Device where the input tensors and sampled alpha values should
            be placed.
        save_state: Whether checkpoints should be written during training.
        save_path: Directory or path prefix used by ``safe_checkpoint``.
        save_name: Base checkpoint name used by ``safe_checkpoint``.
        rank: Distributed process rank. Rank 0 handles logging and checkpointing.
        loss_history: Mutable sequence that is updated in-place with scalar
            training loss values.
        epoch: Current epoch index.
        start_batch: Batch index to resume from. Batches before this index are
            skipped.
        print_every: Logging and checkpointing interval, measured in batches.

    Returns:
        The same ``loss_history`` object, updated with loss values from this run.
    """
    size = len(dataloader.dataset)
    model.train()

    # Some samplers, especially DistributedSampler, need the current epoch to
    # produce a deterministic but epoch-specific shuffle order. The hasattr check
    # keeps this compatible with regular samplers that do not implement set_epoch.
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    for batch, (X, _) in enumerate(dataloader):
        # When resuming from a checkpoint, skip all batches that were already
        # processed in the previous run. This preserves the original batch index
        # values for logging and checkpoint naming.
        if batch < start_batch:
            continue

        X = X.to(device, non_blocking=True)
        batch_size = X.size(0)

        optimizer.zero_grad(set_to_none=True)
         # get_base_model unwraps the trainable model
        base_model = get_base_model(model)

        with torch.no_grad():
            alphas = base_model.sample_alpha(batch_size, device=device)
            original_scores = base_model.valence_score(X)
            target_score = original_scores + alphas

        # The model is trained to produce transformed scores equal to the original
        # scores shifted by alpha. Only the transformed_scores path contributes
        # gradients during backpropagation.
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