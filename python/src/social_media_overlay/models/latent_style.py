"""Latent style-space model with one global learnable edit direction.

This module defines ``LatentStyleModel``, a lightweight baseline for
regressor-guided image editing in MUNIT style space.

The model learns a single global direction ``w`` in flattened style space. Given
an encoded style vector ``S`` and an edit strength ``alpha``, the model applies:

    S_transformed = S + alpha * w

The transformed style is then renormalized to preserve the original style norm,
decoded with a frozen generator, and evaluated with a frozen emotion regressor.

Compared with ``LatentStyleDenseModel``:
    ``LatentStyleModel`` learns one global direction shared by all samples.

    ``LatentStyleDenseModel`` uses a dense direction network to predict a
    sample-dependent direction from each input style vector.

References:
    Goetschalckx et al., "GANalyze: Toward Visual Definitions of Cognitive
    Image Properties", 2019.
    https://arxiv.org/abs/1906.10112

    Official GANalyze implementation:
    https://github.com/LoreGoetschalckx/GANalyze

    Gebhardt et al., regressor-guided image editing implementation:
    https://github.com/christophgebhardt/regressor-guided-image-editing/tree/main/src

    Gebhardt et al., paper:
    https://arxiv.org/pdf/2501.12289
"""

from typing import Optional, Union

import torch.nn as nn
import torch

from social_media_overlay.models.alphas import sample_alpha
from social_media_overlay.models.regressor_utils import valence_score

# Flattened style latent dimensionality used by the MUNIT model in Gebhardt et al.
# For the referenced MUNIT setup, the style code has shape [B, 8, 1, 1], which
# gives a flattened style dimensionality of 8.
DEFAULT_STYLE_DIM = 8

# Minimum alpha value used for regressor-guided editing.
# Alpha controls how far the style vector is moved along the learned direction.
DEFAULT_ALPHA_MIN = -0.5

# Maximum alpha value used for regressor-guided editing.
# Together with DEFAULT_ALPHA_MIN, this defines the uniform sampling range for
# edit strengths during training.
DEFAULT_ALPHA_MAX = 0.5

# Small epsilon used when normalizing transformed style vectors.
# This prevents division by zero if the transformed vector norm becomes very
# small or exactly zero.
NORMALIZE_EPS = 1e-8


class LatentStyleModel(nn.Module):
    """Learn one global linear direction in latent style space.

    This model modifies MUNIT-style latent style codes in order to control an
    image attribute, such as valence, while keeping both the generator and the
    emotion regressor frozen.

    The model is the simpler global-direction counterpart to
    ``LatentStyleDenseModel``. Instead of predicting a sample-dependent direction
    with a dense network, this class learns a single parameter vector ``w`` and
    applies the same direction to every sample in the batch.

    Difference from ``LatentStyleDenseModel``:
        ``LatentStyleModel`` learns one global direction ``w`` with shape
        ``[1, style_dim]``. Every style vector is shifted along this same
        direction.

        ``LatentStyleDenseModel`` replaces this global direction with a dense
        direction network, ``direction_net``, which predicts a separate direction
        for each input style vector.

        In short:

            LatentStyleModel:
                transformed_style = style + alpha * w

            LatentStyleDenseModel:
                direction = direction_net(style)
                transformed_style = style + alpha * direction

    Args:
        regressor: Pretrained emotion regressor. It is expected to be a PyTorch
            model with a ``predict`` method. The first output channel is assumed
            to represent valence.
        generator: Pretrained generator with ``autoencoder_a.encode`` and
            ``autoencoder_a.decode`` methods.
        style_dim: Flattened dimensionality of the style representation.
        a_min: Lower bound for uniformly sampled alpha values.
        a_max: Upper bound for uniformly sampled alpha values.

    Attributes:
        regressor: Frozen pretrained emotion regressor.
        generator: Frozen pretrained generator.
        style_dim: Flattened dimensionality of the style representation.
        loss_func: Mean squared error loss used during training.
        a_min: Minimum alpha value used for scaling the direction.
        a_max: Maximum alpha value used for scaling the direction.
        w: Learnable global direction vector in style space with shape
            ``[1, style_dim]``.
    """

    def __init__(
        self,
        regressor: nn.Module,
        generator: nn.Module,
        style_dim: int = DEFAULT_STYLE_DIM,
        a_min: float = DEFAULT_ALPHA_MIN,
        a_max: float = DEFAULT_ALPHA_MAX,
    ):
        """Initialize the latent style manipulation model.

        The pretrained generator and regressor are stored in evaluation mode and
        frozen. Only the global style-space direction ``w`` is intended to be
        trained.

        Args:
            regressor: Pretrained emotion regressor used for scoring generated
                images. Expected to expose a ``predict`` method.
            generator: Pretrained generator with ``autoencoder_a.encode`` and
                ``autoencoder_a.decode`` methods.
            style_dim: Flattened dimensionality of the style representation.
            a_min: Lower bound for uniformly sampled alpha values.
            a_max: Upper bound for uniformly sampled alpha values.
        """

        super(LatentStyleModel, self).__init__()

        self.regressor = regressor
        self.regressor.eval()

        self.generator = generator
        self.generator.eval()

        # Freeze regressor
        for p in self.regressor.model.parameters():
            p.requires_grad = False

        # Freeze generator
        for p in self.generator.parameters():
            p.requires_grad = False

        self.style_dim = style_dim
        self.loss_func = nn.MSELoss()

        self.a_min = a_min
        self.a_max = a_max

        # Learnable direction in style space
        self.w = nn.Parameter(torch.randn(1, self.style_dim))

    def valence_score(self, images: torch.Tensor) -> torch.Tensor:
        """Predict valence scores for a batch of images.

        The regressor is expected to return at least one output channel, where
        the first channel corresponds to valence. Any additional channels, such
        as arousal, are discarded.

        Args:
            images: Input image batch with shape ``[B, C, H, W]``.

        Returns:
            Valence scores with shape ``[B, 1]``.
        """
        return valence_score(self.regressor, images)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images into generator content and style representations.

        Args:
            images: Input image batch with shape ``[B, C, H, W]``.

        Returns:
            A tuple ``(content, style)`` containing the encoded content and style
            representations produced by ``generator.autoencoder_a.encode``.
        """
        return self.generator.autoencoder_a.encode(images)

    def decode(self, contents: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        """Decode content and style representations into images.

        Args:
            contents: Content representation produced by
                ``generator.autoencoder_a.encode``.
            styles: Style representation produced by
                ``generator.autoencoder_a.encode`` or by
                ``transform_styles``.

        Returns:
            Decoded image batch, usually with shape ``[B, C, H, W]``.
        """
        return self.generator.autoencoder_a.decode(contents, styles)

    def sample_alpha(
        self, batch_size: int, device: Optional[Union[torch.device, str]] = None
    ) -> torch.Tensor:
        """
        Sample alpha values uniformly from the configured range.

        Args:
            batch_size (int): Number of alpha values to sample.
            device (torch.device | None): Target device. If None, uses the
                model device.

        Returns:
            torch.Tensor: Alpha tensor of shape [B, 1].
        """
        if device is None:
            device = next(self.parameters()).device
        return sample_alpha(
            batch_size=batch_size, a_min=self.a_min, a_max=self.a_max, device=device
        )

    def forward(self, X: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Transform image styles and return valence scores.

        The input images are encoded into content and style representations. The
        style representation is flattened, moved along the learned global
        direction ``w``, reshaped back to its original style shape, decoded into
        images, and finally scored by the frozen regressor.

        Args:
            X: Input image batch with shape ``[B, C, H, W]``.
            alphas: Transformation strengths with shape ``[B, 1]``.

        Returns:
            Valence scores of the transformed images with shape ``[B, 1]``.

        Notes:
            The encoder step is wrapped in ``torch.no_grad()`` because the
            generator encoder is frozen.

            The decoder and regressor call are not wrapped in ``torch.no_grad()``
            during the forward pass. This allows gradients to flow back through
            the frozen decoder and frozen regressor into the trainable direction
            parameter ``w``. The frozen modules keep ``requires_grad=False`` for
            their parameters, so their weights are not updated.
        """
        with torch.no_grad():
            batch_size = X.size(0)

            original_content, original_style = self.encode(images=X)
            original_style_flat = original_style.view(batch_size, -1)

        # Transform style
        transformed_style_flat = self.transform_styles(
            S=original_style_flat, alphas=alphas
        )

        # Reshape back to original style shape
        transformed_style = transformed_style_flat.view_as(original_style)

        # Decode transformed image
        transformed_images = self.decode(
            contents=original_content, styles=transformed_style
        )

        transformed_images_score = self.valence_score(transformed_images)
        return transformed_images_score

    def transform_styles(
        self,
        S: torch.Tensor,
        alphas: torch.Tensor,
        normalize_min_cap: float = NORMALIZE_EPS,
    ) -> torch.Tensor:
        """Apply the global style-space direction and renormalize the result.

        Each flattened style vector is shifted along the same learned direction
        ``w``. The result is then renormalized to preserve the original style
        vector norm.

        Compared with ``LatentStyleDenseModel.transform_styles``, this method
        does not predict a direction from the input style. It always uses the
        same global direction parameter ``w``.

        Args:
            S: Flattened style tensor with shape ``[B, D]``.
            alphas: Transformation strengths with shape ``[B, 1]``.
            normalize_min_cap: Minimum denominator used during norm
                renormalization to avoid division by zero.

        Returns:
            Transformed and norm-preserved style tensor with shape ``[B, D]``.
        """
        interim = alphas * self.w
        styles_transformed = S + interim

        # Preserve original norm
        styles_norm = S.norm(dim=1, keepdim=True)
        styles_transformed_norm = styles_transformed.norm(dim=1, keepdim=True).clamp(
            min=normalize_min_cap
        )

        styles_transformed_normalized = (
            styles_norm * styles_transformed / styles_transformed_norm
        )
        return styles_transformed_normalized

    def generate_transformed_images(
        self, X: torch.Tensor, alphas: torch.Tensor
    ) -> torch.Tensor:
        """Generate transformed images without computing regressor scores.

        This method is intended for inference or visualization. It applies the
        same style transformation as ``forward`` but returns the transformed
        images directly instead of passing them through the regressor.

        Args:
            X: Input image batch with shape ``[B, C, H, W]``.
            alphas: Transformation strengths with shape ``[B, 1]``.

        Returns:
            Transformed image batch, usually with shape ``[B, C, H, W]``.
        """
        batch_size = X.size(0)

        original_content, original_style = self.encode(images=X)
        original_style_flat = original_style.view(batch_size, -1)

        transformed_style_flat = self.transform_styles(
            S=original_style_flat, alphas=alphas
        )

        transformed_style = transformed_style_flat.view_as(original_style)

        transformed_images = self.decode(
            contents=original_content, styles=transformed_style
        )

        return transformed_images

    def train(self, mode: bool = True) -> "LatentStyleModel":
        """Set this model's training mode while keeping frozen components in eval mode.

        This overrides ``nn.Module.train`` because the model contains pretrained
        components that must never be switched back to training mode.

        The direction network follows the requested mode, but the generator and
        regressor are always kept in ``eval`` mode.

        Args:
            mode: Whether to set train mode. ``True`` enables training mode for the
                trainable parts of this model, while ``False`` is equivalent to
                calling ``eval``.

        Returns:
            The current model instance.
        """
        # Let PyTorch set the requested mode recursively first.
        # This will also temporarily affect regressor and generator
        super().train(mode)

        # The pretrained components are used only as fixed feature/image mappings.
        # They must stay in eval mode so layers such as BatchNorm and Dropout do not
        # change behavior during training.
        self.regressor.eval()
        self.generator.eval()

        return self

    def eval(self) -> "LatentStyleModel":
        """Set the full model to evaluation mode.

        This keeps the regressor and generator in eval mode, as required, and also
        switches the trainable direction network to eval mode.

        Returns:
            The current model instance.
        """
        return self.train(False)
