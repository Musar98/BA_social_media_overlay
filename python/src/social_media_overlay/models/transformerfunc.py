import torch
import torch.nn as nn

# https://github.com/LoreGoetschalckx/GANalyze/blob/master/pytorch/transformations/pytorch.py


class SingleDirectionTransformer(nn.Module):
    """
    Transformer module that moves latent vectors along one learnable direction.

    This module learns a single direction vector `w` in latent space. During
    transformation, an input latent vector `z` can be shifted along this
    direction by a chosen step size. The shifted vector is then usually
    renormalized so that it keeps the same norm as the original latent vector.

    This type of transformer is commonly used for latent space manipulation,
    where movement along one direction is expected to correspond to one
    semantic change.

    Based on:
        https://github.com/LoreGoetschalckx/GANalyze/blob/master/pytorch/transformations/pytorch.py

    Attributes:
        dim_z (int): Dimensionality of the latent space.
        w (nn.Parameter): Learnable direction vector with shape `(1, dim_z)`.
        criterion (nn.MSELoss): Mean squared error loss used during training.
    """

    def __init__(self, dim_z: int) -> None:
        """
        Initialize the transformer.

        Parameters
        dim_z
            Dimensionality of the latent space.
        """

        super(SingleDirectionTransformer, self).__init__()

        # Dimension of the latent vector
        self.dim_z = dim_z

        # Learnable direction vector in latent space
        # Shape: (1, dim_z)
        self.w = nn.Parameter(torch.randn(1, self.dim_z))

        # Loss function used for training
        self.criterion = nn.MSELoss()

    def transform(self, z: torch.Tensor, step_sizes: torch.Tensor) -> torch.Tensor:
        """
        Transform latent vectors by moving them along the learned direction.

        Each latent vector in `z` is shifted by `step_sizes * self.w`. The result is
        then renormalized per sample so that every transformed vector keeps the same
        L2 norm as its corresponding original latent vector.

        Args:
            z (Tensor): Input latent vectors with shape `(batch_size, dim_z)`.
            step_sizes (Tensor): Scaling factors controlling how far each latent
                vector moves along the learned direction. This should be
                broadcastable to shape `(batch_size, dim_z)`, for example
                `(batch_size, dim_z)`, `(batch_size, 1)`, or `(1, dim_z)`.

        Returns:
            Tensor: Transformed latent vectors with shape `(batch_size, dim_z)`.

        Raises:
            ValueError: If `z` is not a 2D tensor.
            ValueError: If the second dimension of `z` does not match `self.dim_z`.
        """
        # input [B,D] -> in case of munit [B,8]
        # step sizes must be therefore -> [B, D] or [B, 8]
        # w must be [8]!!!!

        # Compute the directional shift
        interim = step_sizes * self.w

        # Apply the shift to the latent vectors
        z_transformed = z + interim

        # Renormalize to keep the same magnitude as the original vector
        # z_transformed = z.norm() * z_transformed / z_transformed.norm() # normalizes batch bug
        # normalize each vector in batch...
        z_norm = z.norm(dim=1, keepdim=True)  # Compute the original vector lengths
        z_transformed_norm = z_transformed.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # clamp to ensure no div by zero

        z_transformed = z_norm * z_transformed / z_transformed_norm  # normalize

        return z_transformed

    def compute_loss(
        self, current: torch.Tensor, target: torch.Tensor, batch_start: int
    ) -> torch.Tensor:
        """
        Compute the training loss between the current output and the target.

        The loss is calculated using mean squared error. The `batch_start`
        argument is currently unused, but is kept in the method signature for
        compatibility with training code that may pass the start index of the
        current batch.

        Args:
            current (Tensor): Current prediction or transformed latent vector.
            target (Tensor): Target latent vector that `current` should approximate.
            batch_start (int): Start index of the current batch in the dataset.
                Currently unused.

        Returns:
            Tensor: Scalar mean squared error loss.
        """

        loss = self.criterion(current, target)

        return loss
