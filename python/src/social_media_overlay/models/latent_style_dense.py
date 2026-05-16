from typing import Optional, Union

import torch
import torch.nn as nn

from social_media_overlay.models.alphas import sample_alpha
from social_media_overlay.models.regressor_utils import valence_score

"""Default hyperparameters for regressor-guided latent direction models.

The constants in this module are used to configure the style-space direction
network and the alpha range for regressor-guided image editing.

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

# Style latent dimensionality used by the MUNIT model in Gebhardt et al.
DEFAULT_STYLE_DIM: int = 8

# Minimum alpha value used for regressor-guided editing.
DEFAULT_ALPHA_MIN: float = (-0.5,)

# Maximum alpha value used for regressor-guided editing.
DEFAULT_ALPHA_MAX: float = 0.5

# Number of linear layers in the direction network.
# This corresponds to the depth of the "transformer function" in GANalyze.
DEFAULT_DIRECTION_DEPTH: int = 2

# Small epsilon used when normalizing latent directions.
# This prevents division by zero when the direction norm is very close to zero.
NORMALIZE_EPS: float = 1e-8


class LatentStyleDenseModel(nn.Module):
    """Learn a sample-dependent style-space transformation.

    This model learns to modify the style representation produced by a frozen
    generator in order to change predicted image emotionality while keeping both
    the generator and the emotion regressor fixed.

    The model follows the general GANalyze idea of moving latent codes along a
    learned direction. However, unlike the original ``SingleDirectionTransformer``,
    which learns one global latent direction ``w`` shared across all samples, this
    model predicts a direction from the current style vector itself. The direction
    is therefore sample-dependent.

    Main differences from ``SingleDirectionTransformer``:
        The original GANalyze transformer learns a single parameter vector
        ``w`` with shape ``[1, dim_z]`` and shifts every latent vector along this
        same global direction.

        This implementation replaces the single learnable direction vector with
        a small dense neural network, ``direction_net``. Given a style vector,
        this network predicts a direction of the same dimensionality.

        The original transformer operates directly on latent vectors ``z``.
        This model operates on MUNIT-style latent style representations produced
        by ``generator.autoencoder_a.encode``.

        The original transformer only defines the latent transformation module.
        This class additionally stores a frozen generator and frozen regressor so
        that transformed styles can be decoded into images and scored by the
        regressor during training.

    Args:
        regressor: Pretrained emotion regressor used to score transformed
            images. It is expected to expose a ``predict`` method and to return
            valence as the first output channel.
        generator: Pretrained generator with ``autoencoder_a.encode`` and
            ``autoencoder_a.decode`` methods.
        style_dim: Flattened dimensionality of the style representation.
        a_min: Lower bound for uniformly sampled alpha values.
        a_max: Upper bound for uniformly sampled alpha values.
        direction_depth: Number of linear layers in the direction network.
            Must be at least ``1``.
        hidden_dim: Hidden width of the direction network. If ``None``, this
            defaults to ``style_dim``.

    Attributes:
        regressor: Frozen pretrained emotion regressor.
        generator: Frozen pretrained image generator.
        style_dim: Flattened dimensionality of the style representation.
        loss_func: Mean squared error loss used during training.
        a_min: Minimum alpha value used for scaling the predicted direction.
        a_max: Maximum alpha value used for scaling the predicted direction.
        direction_net: Dense network that predicts a style-space direction from
            a style vector.

    References:
        Goetschalckx et al., "GANalyze: Toward Visual Definitions of Cognitive
        Image Properties", 2019.
        https://arxiv.org/abs/1906.10112

        Official GANalyze implementation:
        https://github.com/LoreGoetschalckx/GANalyze

        Original ``SingleDirectionTransformer`` implementation:
        https://github.com/LoreGoetschalckx/GANalyze/blob/master/pytorch/transformations/pytorch.py

        Gebhardt et al., regressor-guided image editing implementation:
        https://github.com/christophgebhardt/regressor-guided-image-editing/tree/main/src
    """

    def __init__(
        self,
        regressor: nn.Module,
        generator: nn.Module,
        style_dim: int = DEFAULT_STYLE_DIM,
        a_min: float = DEFAULT_ALPHA_MIN,
        a_max: float = DEFAULT_ALPHA_MAX,
        direction_depth: int = DEFAULT_DIRECTION_DEPTH,
        hidden_dim: Optional[int] = None,
    ):
        """Initialize the latent style transformation model.

        The pretrained generator and regressor are stored in evaluation mode and
        frozen. Only the dense direction network is intended to be trained.

        Args:
            regressor: Pretrained emotion regressor used for scoring transformed
                images. Expected to expose a ``predict`` method and return
                valence as the first output channel.
            generator: Pretrained generator with ``autoencoder_a.encode`` and
                ``autoencoder_a.decode`` methods.
            style_dim: Flattened dimensionality of the style representation.
            a_min: Lower bound for uniformly sampled alpha values.
            a_max: Upper bound for uniformly sampled alpha values.
            direction_depth: Number of linear layers in the direction network.
                Must be at least ``1``.
            hidden_dim: Hidden width of the direction network. If ``None``,
                defaults to ``style_dim``.

        Raises:
            ValueError: If ``direction_depth`` is smaller than ``1``.
        """
        super().__init__()

        self.regressor = regressor
        self.regressor.eval()

        self.generator = generator
        self.generator.eval()

        # Freeze pretrained components.
        for p in self.regressor.model.parameters():
            p.requires_grad = False

        for p in self.generator.parameters():
            p.requires_grad = False

        self.style_dim = style_dim
        self.loss_func = nn.MSELoss()
        self.a_min = a_min
        self.a_max = a_max

        if direction_depth < 1:
            raise ValueError("direction_depth must be >= 1")

        if hidden_dim is None:
            hidden_dim = style_dim

        self.direction_net = self._build_direction_net(
            in_dim=style_dim,
            out_dim=style_dim,
            hidden_dim=hidden_dim,
            depth=direction_depth,
        )

    def _build_direction_net(
        self, in_dim: int, out_dim: int, hidden_dim: int, depth: int
    ) -> nn.Sequential:
        """Build the dense network that predicts a style-space direction.

        The network maps a flattened style vector to a direction vector of the same
        or another requested dimensionality. This replaces the single global
        direction parameter used in the original GANalyze ``SingleDirectionTransformer``
        with a sample-dependent direction predictor.

        Network structure:
            If ``depth == 1``, the network is a single linear projection from
            ``in_dim`` to ``out_dim``.

            If ``depth > 1``, the network consists of an input linear layer,
            ``depth - 2`` hidden linear layers, and a final output linear layer.
            ReLU activations are applied after all linear layers except the final
            output layer.

        Args:
            in_dim: Dimensionality of the input style vector.
            out_dim: Dimensionality of the predicted direction vector.
            hidden_dim: Width of the hidden layers.
            depth: Number of linear layers in the network. Must be at least ``1``.

        Returns:
            A ``torch.nn.Sequential`` module that maps tensors of shape
            ``[B, in_dim]`` to tensors of shape ``[B, out_dim]``.

        Raises:
            ValueError: If ``depth`` is smaller than ``1``.
        """
        layers = []

        if depth == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, out_dim))

        return nn.Sequential(*layers)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images into content and style representations.
        """
        return self.generator.autoencoder_a.encode(images)

    def decode(self, contents: torch.Tensor, styles: torch.tensor) -> torch.Tensor:
        """Decode content and style representations into images.

        The generator is assumed to be frozen and kept in evaluation mode, so this method
        only uses the generator as a fixed decoder.

        Args:
            contents: Content representation produced by
                ``generator.autoencoder_a.encode``.
            styles: Style representation produced by
                ``generator.autoencoder_a.encode`` or a transformed style
                representation produced by the direction network.

        Returns:
            Decoded image batch as a tensor. The expected shape is usually
            ``[B, C, H, W]``, depending on the generator output format.

        Raises:
            AttributeError: If ``generator`` does not provide
                ``autoencoder_a.decode``.
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
        """Predict valence scores after applying a style-space transformation.

        The input images are first encoded into content and style representations
        using the frozen generator encoder. The style representation is flattened,
        transformed by the trainable direction network, reshaped back to its original
        style shape, decoded into images, and finally scored by the frozen emotion
        regressor.

        Args:
            X: Input image batch with shape ``[B, C, H, W]``.
            alphas: Transformation strengths with shape ``[B, 1]``. Each alpha
                controls how far the corresponding image's style representation is
                moved along its predicted style-space direction.

        Returns:
            Valence predictions for the transformed images with shape ``[B, 1]``.

        Notes:
            The original content and style encodings are computed under
            ``torch.no_grad()`` because the generator encoder is frozen and does not
            need gradients.

            The transformed style, decoder output, and regressor score are not
            wrapped in ``torch.no_grad()``. This allows gradients to flow from the
            valence loss back through the frozen regressor and frozen generator
            decoder into the trainable direction network. The frozen modules keep
            ``requires_grad=False`` for their parameters, so their weights are not
            updated.

            The generator and regressor should remain in ``eval`` mode throughout
            training. The class-level ``train``/``eval`` overrides are responsible
            for enforcing that behavior.

        Raises:
            RuntimeError: If the flattened style dimensionality does not match the
                expected input dimensionality of ``transform_styles``.
            ValueError: If ``alphas`` has a shape that cannot broadcast against the
                flattened style representation.
        """
        with torch.no_grad():
            batch_size = X.size(0)
            original_content, original_style = self.encode(images=X)
            original_style_flat = original_style.view(batch_size, -1)

        transformed_style_flat = self.transform_styles(
            z=original_style_flat, alphas=alphas
        )

        transformed_style = transformed_style_flat.view_as(original_style)

        transformed_images = self.decode(
            contents=original_content, styles=transformed_style
        )

        transformed_images_score = valence_score(self.regressor, transformed_images)
        return transformed_images_score

    def transform_styles(
        self,
        z: torch.Tensor,
        alphas: torch.Tensor,
        normalize_min_cap: float = NORMALIZE_EPS,
    ) -> torch.Tensor:
        """Predict and apply a sample-dependent style-space direction.

        The direction network predicts one direction vector for each flattened style
        vector. Each style vector is then shifted along its predicted direction using
        the corresponding alpha value. Finally, the transformed style vector is
        renormalized to have the same norm as the original style vector.

        This follows the GANalyze-style idea of moving a latent code along a learned
        direction, but differs from the original ``SingleDirectionTransformer`` in
        one important way: the original transformer uses one global learnable
        direction shared by all samples, while this method uses ``direction_net`` to
        predict a direction conditioned on each input style vector.

        Args:
            z: Flattened style tensor with shape ``[B, D]``, where ``B`` is the batch
                size and ``D`` is the flattened style dimensionality.
            alphas: Transformation strengths with shape ``[B, 1]``. Each alpha
                controls the step size for the corresponding style vector.
            normalize_min_cap: Minimum value used when dividing by the transformed
                style norm. This avoids division by zero or unstable division by a
                very small norm.

        Returns:
            Transformed flattened style tensor with shape ``[B, D]``.

        Raises:
            RuntimeError: If ``z`` is incompatible with ``direction_net`` or if
                ``alphas`` cannot broadcast against the predicted direction.
        """
        direction: torch.Tensor = self.direction_net(z)
        z_transformed = z + alphas * direction

        z_norm = z.norm(dim=1, keepdim=True)
        z_transformed_norm = z_transformed.norm(dim=1, keepdim=True).clamp(
            min=normalize_min_cap
        )
        z_transformed = z_norm * z_transformed / z_transformed_norm

        return z_transformed

    def generate_transformed_images(
        self, X: torch.Tensor, alphas: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate transformed images without returning regressor scores.

        Args:
            X (torch.Tensor): Input images of shape [B, C, H, W].
            alphas (torch.Tensor): Transformation strengths of shape [B, 1].

        Returns:
            torch.Tensor: Transformed images of shape [B, C, H, W].
        """
        with torch.no_grad():
            batch_size = X.size(0)
            original_content, original_style = self.encode(images=X)
            original_style_flat = original_style.view(batch_size, -1)

            transformed_style_flat = self.transform_styles(
                z=original_style_flat, alphas=alphas
            )

            transformed_style = transformed_style_flat.view_as(original_style)

            transformed_images = self.decode(
                contents=original_content, styles=transformed_style
            )

        return transformed_images

    def train(self, mode: bool = True) -> "LatentStyleDenseModel":
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

        # Only the learnable direction network should follow the requested mode.
        self.direction_net.train(mode)

        return self

    def eval(self) -> "LatentStyleDenseModel":
        """Set the full model to evaluation mode.

        This keeps the regressor and generator in eval mode, as required, and also
        switches the trainable direction network to eval mode.

        Returns:
            The current model instance.
        """
        return self.train(False)

    def valence_score(self, images: torch.Tensor) -> torch.Tensor:
        return valence_score(self.regressor, images)
