import torch
import torch.nn as nn


class Multiply(nn.Module):
    """
    Scale a tensor by a fixed constant.

    This small module is used inside parameter prediction heads so output-range
    scaling stays visible as part of the head definition. For example, a head may
    use `Sigmoid()` followed by `Multiply(3.0)` to produce values in roughly the
    range [0, 3].

    Args:
        alpha:
            Fixed scalar multiplier applied to the input tensor.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply the input tensor by the stored scalar.

        Args:
            x:
                Input tensor of any shape.

        Returns:
            Tensor with the same shape as `x`, scaled by `self.alpha`.
        """
        return x * self.alpha
