from typing import Any

import torch


def get_first_n_images(dataloader: torch.utils.data.DataLoader, n: int = 8):
    """
    Extract the first `n` images from a PyTorch DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding batches. Each batch
            can be either a tensor of images or a tuple/list where the first
            element contains the images, for example `(images, labels)`.
        n (int, optional): Number of images to retrieve. Defaults to 8.

    Returns:
        Tensor: Tensor of shape `(k, ...)`, where `k <= n`, containing the
        collected images stacked along the first dimension.

    Raises:
        RuntimeError: If no images are found in the dataloader.
    """
    imgs = []

    for batch in dataloader:
        # Support common dataloader outputs such as (images, labels)
        images = batch[0] if isinstance(batch, (list, tuple)) else batch

        for img in images:
            imgs.append(img)

            # Early exit once enough images are collected
            if len(imgs) == n:
                return torch.stack(imgs)

    # Handle case where fewer than n images are available
    if not imgs:
        raise RuntimeError("No images found in dataloader.")

    return torch.stack(imgs)


def to_tensor(
    x: Any,
    device: torch.types.Device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert input data to a PyTorch tensor on the given device and with the given dtype.

    If `x` is already a tensor, it is moved or cast using `.to(...)`.
    Otherwise, a new tensor is created from `x`.

    Args:
        x (Any): Input data to convert. Can be a tensor, list, tuple, NumPy array,
            scalar, or other tensor-compatible object.
        device (Device): Target device, for example `"cpu"`, `"cuda"`, or
            `torch.device("cuda")`.
        dtype (torch.dtype): Target tensor dtype, for example `torch.float32`.

    Returns:
        Tensor: Tensor representation of `x` on the given device and with the
        given dtype.
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def zero_scalar(device: torch.types.Device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a scalar tensor containing zero on the given device and with the given dtype.

    Args:
        device (Device): Target device, for example `"cpu"`, `"cuda"`, or
            `torch.device("cuda")`.
        dtype (torch.dtype): Target tensor dtype, for example `torch.float32`.

    Returns:
        Tensor: Scalar tensor containing `0.0` on the given device and with the
        given dtype.
    """
    return torch.tensor(0.0, device=device, dtype=dtype)
