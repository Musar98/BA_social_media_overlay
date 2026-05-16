import os
import torch
import torch.distributed as dist
from typing import Any


def setup_ddp() -> tuple[int, int, int]:
    """
    Set up PyTorch Distributed Data Parallel (DDP).

    Reads rank information from environment variables (set by torchrun),
    assigns the correct GPU, and initializes the process group.

    Returns:
        rank: global process ID
        local_rank: GPU index on this node
        world_size: total number of processes
    """

    rank = int(os.environ["RANK"])  # unique ID across all processes
    local_rank = int(os.environ["LOCAL_RANK"])  # GPU index on current machine
    world_size = int(os.environ["WORLD_SIZE"])  # total number of processes

    # Bind this process to the correct GPU
    torch.cuda.set_device(local_rank)

    # Initialize communication backend (NCCL = fast GPU comms)
    dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    """
    Clean up the Distributed Data Parallel (DDP) process group.

    Safely shuts down the distributed backend if it was initialized.
    Should be called at the end of training to free resources.
    """

    # Only attempt cleanup if DDP was initialized
    if dist.is_initialized():
        # Tear down the process group and release communication resources
        dist.destroy_process_group()


def get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying model, unwrapping DDP/DataParallel if needed.

    Args:
        model: A PyTorch model, possibly wrapped in DDP or DataParallel.

    Returns:
        The original (unwrapped) model.
    """

    # DDP/DataParallel wrap the model inside `.module`
    # If present, return the inner model, otherwise return as-is
    return model.module if hasattr(model, "module") else model


def broadcast_object(obj: object, rank: int, src: int = 0) -> object:
    """
    Broadcast a Python object from the source rank to all processes.

    Args:
        obj: Object to broadcast (only used on source rank).
        rank: Rank of the current process.
        src: Source rank that provides the object (default: 0).

    Returns:
        The broadcasted object (same on all ranks).
    """

    # Only the source rank provides the object, others start with None
    obj_list = [obj if rank == src else None]

    # Broadcast the object to all processes
    dist.broadcast_object_list(obj_list, src=src)

    # After broadcast, all ranks have the same object
    return obj_list[0]


def load_checkpoint_if_requested(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    rank: int,
    checkpoint_path: str,
    load_optimizer: bool = True,
):
    """
    Load a training checkpoint if a checkpoint path was provided.

    In a DDP setup, only rank 0 checks and validates the checkpoint path.
    The selected checkpoint path is then broadcast to all other ranks so that
    every process loads the same checkpoint state.

    If no checkpoint path is provided, training starts from scratch.

    Args:
        model: Model whose parameters should be restored. May be a plain
            PyTorch module or wrapped in DistributedDataParallel/DataParallel.
        optimizer: Optimizer whose state should optionally be restored.
        device: Device used for loading the checkpoint, for example "cuda:0",
            "cpu", or a torch.device.
        rank: Global DDP rank of the current process.
        checkpoint_path: Path to the checkpoint file. If this is an empty
            string, no checkpoint is loaded.
        load_optimizer: If True, restore the optimizer state from the
            checkpoint. If False, only the model state is restored.

    Returns:
        A tuple containing:
            start_epoch: Epoch to continue training from. Returns 0 for a new run.
            start_batch: Batch index to continue from. Returns 0 for a new run.
            loss_history: Previously stored training loss values, or an empty list.
            val_loss_history: Previously stored validation loss values, or an empty list.

    Raises:
        FileNotFoundError: If rank 0 receives a non-empty checkpoint path that
            does not exist.
        KeyError: If the checkpoint is missing required keys such as
            "model_state_dict", "epoch", or "batch".
        RuntimeError: If loading the model or optimizer state fails.
    """
    if rank == 0:
        ckpt_to_load = checkpoint_path.strip()
        if ckpt_to_load == "":
            print("No checkpoint provided, starting new training run.")
            payload = None
        else:
            if not os.path.isfile(ckpt_to_load):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_to_load}")
            print(f"Loading checkpoint: {ckpt_to_load}")
            payload = ckpt_to_load
    else:
        payload = None

    payload = broadcast_object(payload, rank, src=0)

    if payload is None:
        return 0, 0, [], []

    checkpoint = torch.load(payload, map_location=device)

    base_model = get_base_model(model)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint["epoch"])
    start_batch = int(checkpoint["batch"]) + 1
    loss_history = checkpoint.get("loss_history", [])
    val_loss_history = checkpoint.get("val_loss_history", [])

    if rank == 0:
        print(
            f"Resumed from epoch={checkpoint['epoch']} batch={checkpoint['batch']}. "
            f"Continuing at epoch={start_epoch}, batch={start_batch}."
        )
        print(f"Loaded training loss history with {len(loss_history)} entries.")
        print(f"Loaded validation loss history with {len(val_loss_history)} entries.")

    return start_epoch, start_batch, loss_history, val_loss_history


def safe_checkpoint(
    save_dir: str | os.PathLike[str],
    save_name: str,
    last_epoch: int,
    last_batch: int,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    loss_history: list[Any],
    val_loss_history: list[Any] = None,
    name_suffix: str = "-final.pt",
) -> None:
    """
    Save a training checkpoint to disk.

    The checkpoint stores the model state, optimizer state, training progress,
    and loss histories in a single file so that training can later be resumed
    or analyzed.

    Args:
        save_dir: Directory where the checkpoint file should be saved.
        save_name: Base name of the checkpoint file, without the suffix.
        last_epoch: Index of the last completed or currently saved epoch.
        last_batch: Index of the last processed batch within the epoch.
        model_state_dict: Model parameters, usually from `model.state_dict()`.
        optimizer_state_dict: Optimizer state, usually from
            `optimizer.state_dict()`.
        loss_history: Training loss history collected so far.
        val_loss_history: Optional validation loss history collected so far.
            Defaults to None if no validation history is available.
        name_suffix: Suffix appended to `save_name` when building the final
            filename. Defaults to "-final.pt".

    Returns:
        None.

    Raises:
        OSError: If the checkpoint path cannot be written.
        RuntimeError: If `torch.save` fails while serializing the checkpoint.
    """

    # Build full file path (e.g., "checkpoints/model-final.pt")
    final_path = os.path.join(save_dir, f"{save_name}{name_suffix}")

    # Save all relevant training state in a single dictionary
    torch.save(
        {
            "epoch": last_epoch,
            "batch": last_batch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "loss_history": loss_history,
            "val_loss_history": val_loss_history,
        },
        final_path,
    )
