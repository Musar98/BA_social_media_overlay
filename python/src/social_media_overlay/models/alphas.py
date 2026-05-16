from typing import Optional, Union

import torch


def sample_alpha(
    batch_size: int,
    a_min: float,
    a_max: float,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    Sample transformation strengths (alpha values).

    Args:
        batch_size (int): Number of samples.
        device (torch.device, optional): Target device.

    Returns:
        torch.Tensor: Alpha values [B, 1] uniformly sampled in [a_min, a_max].
    """
    return torch.empty(batch_size, 1, device=device).uniform_(a_min, a_max)
