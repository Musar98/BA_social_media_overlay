import torch


def sample_alphas(
    batch_size: int,
    device: torch.device,
    zero_alpha_prob: float = 0.2,
    low: float = -0.5,
    high: float = 0.5,
) -> torch.Tensor:
    """
    Sample alpha values for a batch, with optional zeros.

    Alphas are drawn uniformly from [low, high], but a fraction
    (zero_alpha_prob) is set exactly to 0.

    Args:
        batch_size: Number of samples.
        device: Torch device (e.g., "cuda").
        zero_alpha_prob: Probability of sampling alpha = 0.
        low: Lower bound of uniform distribution.
        high: Upper bound of uniform distribution.

    Returns:
        Tensor of shape (batch_size, 1) containing alpha values.
    """

    # Sample uniformly in the given range
    alphas = torch.empty(batch_size, 1, device=device).uniform_(low, high)

    # Create mask for values that should be forced to zero
    zero_mask = torch.rand(batch_size, 1, device=device) < zero_alpha_prob

    # Set selected entries to zero
    alphas[zero_mask] = 0.0

    return alphas