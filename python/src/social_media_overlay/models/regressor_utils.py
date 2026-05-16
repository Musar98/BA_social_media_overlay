import torch


def valence_score(regressor: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Predict valence scores for a batch of images.

    The regressor is a PyTorch model that provides a ``predict`` method. The
    first output channel is assumed to contain valence. Any additional channels,
    such as arousal, are discarded.

    Args:
        regressor: PyTorch emotion regressor. Its ``predict`` method must accept
            an image tensor with shape ``[B, C, H, W]`` and return predictions
            with shape ``[B, N]``, where ``N >= 1``.
        images: Input image batch with shape ``[B, C, H, W]``.

    Returns:
        A tensor with shape ``[B, 1]`` containing one valence score per image.
    """

    # Keep only the first output channel, which is assumed to represent valence.
    # Using :1 instead of 0 preserves the two-dimensional shape [B, 1].
    return regressor.predict(images)[:, :1]
