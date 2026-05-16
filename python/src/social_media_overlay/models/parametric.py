"""
Refactored conditional parametric image generators.

This file contains the newer generator variants. The refactor separates encoder construction,
MobileNet setup, preprocessing, freezing, condition encoding, parameter
prediction, and parameter flattening into clearer helper functions and model
classes.

Included models:

    ConditionalParametricConvGenerator
        Conv-only baseline. Uses a learned convolutional encoder and directly
        predicts the final transform parameters passed to apply_params.

    ConditionalParametricMobileNetGenerator
        MobileNet-only baseline. Uses a timm MobileNet backbone for image
        features and directly predicts final transform parameters. Alpha is
        encoded and concatenated with image features, but alpha=0 is not
        guaranteed to mean identity.

    ConditionalParametricResidualMobileNetGenerator
        Identity-aware MobileNet model. Predicts raw residual values instead of
        final parameters, combines them with alpha, and maps them into bounded
        valid parameter ranges around explicit identity values. This is designed
        so alpha=0 stays close to no edit.

    MultiConditionalParametricResidualMobileNetGenerator
        Multi-condition version of the residual MobileNet model. It accepts
        multiple conditioning values, derives an internal condition signal, and
        uses it to modulate residual parameter predictions.

Common input/output contract:

    - x is an image tensor in [0, 1], usually [B, C, H, W].
    - alpha controls the edit strength or condition.
    - single-condition models accept alpha as [B] or [B, 1].
    - the multi-condition model expects a multi-value alpha, currently [B, 2].
    - forward returns the final transformed image and flattened parameters.

The standard transformation order is defined by TRANSFORMATIONS_IN_ORDER and includes:

    exposure, saturation, tone, color, contrast, sharp, blur

The absolute models, ConditionalParametricConvGenerator and
ConditionalParametricMobileNetGenerator, predict transform values directly:

    params = head(MLP(image_features, alpha_condition))

The residual models, ConditionalParametricResidualMobileNetGenerator and
MultiConditionalParametricResidualMobileNetGenerator, predict residuals around
identity:

    raw = head(latent)
    u = condition * raw
    param = transform_param(u, min_val, max_val, identity)

Use the absolute models as simpler baselines. Use the residual models when alpha
should behave like a controllable edit strength and the neutral condition should
produce little or no change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from social_media_overlay.ext.gebhardt.image_transformations.image_transformations import (
    apply_params,
)
from social_media_overlay.models.film import FiLMResidualBlock
from social_media_overlay.models.multiply import Multiply
from social_media_overlay.models.parametric_utils import (
    COLOR_CHANNELS,
    CURVE_STEPS,
    PARAMS,
    build_cond_encoder,
    build_conv_encoder,
    create_mobilenet_backbone,
    flatten_output_params,
    freeze_mobilenet_backbone,
    infer_mobilenet_encoder_dim,
    normalize_for_mobilenet,
    piecewise_tanh_mapping,
    resolve_backbone_stats_and_input_size,
    shifted_sigmoid_mapping,
    validate_mapping_bounds,
)


class ConditionalParametricConvGenerator(nn.Module):
    """
    OLDER VARIANT!
    Conditional parametric image generator with a plain convolutional encoder.

    This is the conv-only baseline model. It encodes the input image with a
    learned convolutional encoder, encodes the scalar condition `alpha` with a
    small MLP, concatenates both feature vectors, and predicts parameters for
    the image transformation pipeline.

    This model uses the absolute parameterization: each prediction head directly
    outputs the final transform value expected by `apply_params`. Therefore,
    `alpha` is an input condition, but `alpha=0` is not guaranteed by the
    architecture to produce the identity transform.

    Expected inputs:
        x:
            Image tensor in [0, 1], shape [B, C, H, W].

        alpha:
            Scalar condition tensor, shape [B] or [B, 1].

    Returns:
        final_img:
            Final transformed image, usually shape [B, 3, H, W].

        flat_params:
            Flattened transform parameters, useful for logging, debugging, or
            auxiliary losses.
    """

    def __init__(
        self,
        num_features: int,
        fc_dim: int,
        num_conv_layers: int = 7,
        do_norm: bool = True,
        in_channels: int = 3,
        cond_dim: int = 32,
    ) -> None:
        """
        Initialize the conv-only conditional parametric generator.

        Args:
            num_features:
                Base number of convolutional channels. The first conv block
                outputs this many channels, and later blocks increase the
                channel count.

            fc_dim:
                Width of the fully connected trunk used after concatenating
                image features and alpha features.

            num_conv_layers:
                Number of convolutional encoder blocks.

            do_norm:
                If True, use InstanceNorm2d inside convolutional blocks.

            in_channels:
                Number of channels in the input image tensor.

            cond_dim:
                Size of the learned alpha condition embedding.
        """
        super().__init__()
        params = PARAMS
        self.num_features = num_features
        self.fc_dim = fc_dim
        self.do_norm = do_norm
        self.cond_dim = cond_dim
        self.curve_steps = CURVE_STEPS
        self.color_channels = COLOR_CHANNELS
        # Stores the intermediate images returned by apply_params.
        # The final output is self.img_list[-1].
        self.img_list = None

        self.conv_layers, encoder_dim = build_conv_encoder(
            in_channels=in_channels,
            num_features=num_features,
            num_conv_layers=num_conv_layers,
            do_norm=do_norm,
        )

        self.cond_encoder = build_cond_encoder(cond_dim)

        # Shared trunk after image features and alpha embedding are concatenated.
        self.dense0 = nn.Linear(encoder_dim + cond_dim, fc_dim)
        self.dense1 = nn.Linear(fc_dim, fc_dim)

        self.param_heads = self.define_parameter_prediction_heads(fc_dim, params)

    def define_parameter_prediction_heads(
        self,
        fc_dim: int,
        params: list[str] | tuple[str, ...],
    ) -> nn.ModuleDict:
        """
        Build parameter prediction heads for the absolute parameterization.

        Each head maps the shared latent vector of shape [B, fc_dim] to one
        image-transformation parameter. The output activations and scaling
        constants are chosen to match the value ranges expected by the
        downstream image transformation functions.

        Most scalar heads produce [B, 1], which is later squeezed to [B].
        Curve heads keep structured shapes because `apply_params` expects them:

            tone:  [B, 1, CURVE_STEPS, 1]
            color: [B, 3, CURVE_STEPS, 1]

        Args:
            fc_dim:
                Input feature dimension for every prediction head.

            params:
                Iterable of parameter names to build heads for.

        Returns:
            ModuleDict mapping parameter names to PyTorch prediction heads.
        """
        param_heads = nn.ModuleDict()

        for param in params:
            head = None

            if param == "sharp":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(15.0)
                )  # (0, 15)
            elif param == "exposure":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(3.0)
                )  # (-3, 3)
            elif param == "contrast":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(3.0)
                )  # (0, 3)
            elif param == "saturation":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(10.0)
                )  # (0, 10)
            elif param == "tone":  # (0, 3)
                head = nn.Sequential(
                    nn.Linear(fc_dim, self.curve_steps),
                    nn.Sigmoid(),
                    nn.Unflatten(1, (1, self.curve_steps, 1)),
                    Multiply(3.0),
                )
            elif param == "color":  # (0, 3)
                head = nn.Sequential(
                    nn.Linear(fc_dim, self.curve_steps * 3),
                    nn.Sigmoid(),
                    nn.Unflatten(1, (3, self.curve_steps, 1)),
                    Multiply(3.0),
                )
            elif param == "blur":  # (0, 10)
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(10.0))

            if head is not None:
                param_heads[param] = head

        return param_heads

    def get_img_enc(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        # Global average pooling. This removes spatial dimensions and keeps
        # only the channel-wise activation summary for each image.
        return x.mean(dim=[2, 3])

    def get_predicted_params(
        self,
        img_features: torch.Tensor,
        alpha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict absolute transformation parameters from image features and alpha.

        Alpha is first encoded into a learned condition vector. This condition
        vector is concatenated with the image feature vector and passed through
        the shared fully connected trunk. The resulting latent vector is then
        sent through one prediction head per transformation parameter.

        Args:
            img_features:
                Encoded image features, shape [B, encoder_dim].

            alpha:
                Scalar condition tensor, shape [B] or [B, 1].

        Returns:
            Dictionary mapping parameter names to tensors accepted by
            `apply_params`.
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        cond = self.cond_encoder(alpha)
        x = torch.cat([img_features, cond], dim=1)

        assert x.shape[1] == self.dense0.in_features, (
            f"Feature mismatch: got {x.shape[1]} features, "
            f"but dense0 expects {self.dense0.in_features}"
        )

        x = F.leaky_relu(self.dense0(x))
        x = F.leaky_relu(self.dense1(x))

        dict_params = {}

        for param, head in self.param_heads.items():
            out: torch.Tensor = head(x)

            # Scalar heads return [B, 1]. Squeeze to [B] to preserve the
            # original output contract expected by apply_params.
            #
            # Curve heads are intentionally not squeezed because they have
            # structured shapes such as [B, 1, CURVE_STEPS, 1].
            if out.ndim == 2 and out.shape[1] == 1:
                out = out.squeeze(1)

            dict_params[param] = out

        return dict_params

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full generator forward pass.

        The model predicts transformation parameters from the image and alpha,
        applies them to the RGB image channels, and returns both the final image
        and flattened parameter values.

        Args:
            x:
                Input image tensor in [0, 1], shape [B, C, H, W].

            alpha:
                Scalar condition tensor, shape [B] or [B, 1].

        Returns:
            A tuple containing:

            final_img:
                Final transformed image, shape [B, 3, H, W].

            flat_params:
                Flattened parameter tensor in the standard parameter order.
        """
        enc = self.get_img_enc(x)
        params = self.get_predicted_params(enc, alpha)

        self.img_list = apply_params(x[:, : self.color_channels, :, :], params)
        flat_params = flatten_output_params(params, x.shape[0])

        return self.img_list[-1], flat_params


class ConditionalParametricMobileNetGenerator(nn.Module):
    """
    OLDER VARIANT!
    Conditional parametric image generator with a timm MobileNet backbone.

    This is the MobileNet-based absolute-parameter baseline. It uses a pretrained
    or randomly initialized timm MobileNet model as an image feature extractor,
    encodes the scalar condition `alpha` with a small MLP, concatenates image and
    condition features, and directly predicts the parameters passed to
    `apply_params`.

    This model uses the absolute parameterization: every prediction head outputs
    the final transform value directly. Alpha is part of the network input, but
    `alpha=0` is not guaranteed by the architecture to produce an identity/no-op
    transformation.

    Expected inputs:
        x:
            Image tensor in [0, 1], shape [B, C, H, W].

        alpha:
            Scalar condition tensor, shape [B] or [B, 1].

    Returns:
        final_img:
            Final transformed image, usually shape [B, 3, H, W].

        flat_params:
            Flattened transform parameters, useful for logging, debugging, or
            auxiliary supervision.
    """

    def __init__(
        self,
        fc_width: int = 128,
        fc_depth: int = 2,
        in_channels: int = 3,
        mobilenet_variant: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        backbone_input_size: int | None = None,
        mobilenet_pretrained: bool = True,
        freeze_backbone: bool = True,
        cond_dim: int = 32,
    ) -> None:
        """
        Initialize the MobileNet-based conditional parametric generator.

        Args:
            fc_width:
                Width of each fully connected layer after image and condition
                features are concatenated.

            fc_depth:
                Number of fully connected layers in the shared prediction trunk.
                Must be at least 1.

            in_channels:
                Number of input image channels expected by the backbone.

            mobilenet_variant:
                Name of the timm MobileNet variant to create.

            backbone_input_size:
                Optional manual square input size for the backbone. If None, the
                size from the timm data config is used.

            mobilenet_pretrained:
                If True, load pretrained timm weights and normalize inputs with
                the timm mean/std. If False, use random weights and identity
                normalization.

            freeze_backbone:
                If True, keep MobileNet fixed as a feature extractor.

            cond_dim:
                Size of the learned alpha condition embedding.
        """
        super().__init__()

        params = PARAMS

        if fc_depth < 1:
            raise ValueError(f"fc_depth must be >= 1, got {fc_depth}")
        if fc_width < 1:
            raise ValueError(f"fc_width must be >= 1, got {fc_width}")

        self.fc_width = fc_width
        self.fc_depth = fc_depth
        self.cond_dim = cond_dim
        self.curve_steps = CURVE_STEPS
        self.color_channels = COLOR_CHANNELS
        self.mobilenet_pretrained = mobilenet_pretrained

        # Stores intermediate images returned by apply_params.
        # The final output is self.img_list[-1].
        self.img_list = None

        # Create MobileNet as a feature-map backbone, not a classifier.
        # create_mobilenet_backbone returns both the model and timm's data config
        # so preprocessing can match the selected variant.
        self.backbone, data_cfg = create_mobilenet_backbone(
            mobilenet_variant=mobilenet_variant,
            in_channels=in_channels,
            mobilenet_pretrained=mobilenet_pretrained,
        )

        # Resolve input size and normalization statistics. Pretrained models use
        # timm mean/std, while non-pretrained models use identity normalization.
        backbone_mean, backbone_std, self.backbone_input_hw = (
            resolve_backbone_stats_and_input_size(
                data_cfg=data_cfg,
                in_channels=in_channels,
                mobilenet_pretrained=mobilenet_pretrained,
                backbone_input_size=backbone_input_size,
            )
        )
        # Kept as a convenience/debugging attribute for square-input backbones.
        self.backbone_input_size = self.backbone_input_hw[0]

        freeze_mobilenet_backbone(self.backbone, freeze_backbone)

        # Infer the MobileNet feature dimension dynamically because different
        # timm variants may output different channel counts.
        encoder_dim = infer_mobilenet_encoder_dim(
            backbone=self.backbone,
            in_channels=in_channels,
            backbone_input_hw=self.backbone_input_hw,
            backbone_mean=backbone_mean,
            backbone_std=backbone_std,
            mobilenet_pretrained=mobilenet_pretrained,
        )

        # Buffers move automatically with model.to(device), but are not trainable.
        # They are used in normalize_for_backbone before MobileNet encoding.
        self.register_buffer(
            "backbone_mean",
            torch.tensor(backbone_mean, dtype=torch.float32).view(
                1, len(backbone_mean), 1, 1
            ),
        )
        self.register_buffer(
            "backbone_std",
            torch.tensor(backbone_std, dtype=torch.float32).view(
                1, len(backbone_std), 1, 1
            ),
        )

        self.cond_encoder = build_cond_encoder(cond_dim)

        # Shared MLP trunk after concatenating image features and alpha features.
        # First layer receives [image_features, condition_embedding].
        fc_layers = []
        in_dim = encoder_dim + cond_dim
        for _ in range(fc_depth):
            fc_layers.append(nn.Linear(in_dim, fc_width))
            in_dim = fc_width

        self.fc_layers = nn.ModuleList(fc_layers)

        self.param_heads = self.define_parameter_prediction_heads(fc_width, params)

    def define_parameter_prediction_heads(
        self,
        fc_dim: int,
        params: list[str] | tuple[str, ...],
    ) -> nn.ModuleDict:
        """
        Build prediction heads for the absolute MobileNet parameterization.

        Each head maps the shared latent tensor of shape [B, fc_dim] to one
        transform parameter. The activation and scaling of each head define the
        operational range of the predicted value.

        Scalar heads usually return [B, 1] and are later squeezed to [B].
        Curve heads keep structured shapes expected by `apply_params`:

            tone:  [B, 1, CURVE_STEPS, 1]
            color: [B, 3, CURVE_STEPS, 1]

        Args:
            fc_dim:
                Input feature dimension for every prediction head.

            params:
                Iterable of parameter names to build heads for.

        Returns:
            ModuleDict mapping parameter names to prediction heads.
        """
        param_heads = nn.ModuleDict()

        for param in params:
            head = None

            if param == "sharp":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(5.0))
            elif param == "exposure":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(0.1))
            elif param == "contrast":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(0.15))
            elif param == "saturation":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(10.0))
            elif param == "tone":
                head = nn.Sequential(
                    nn.Linear(fc_dim, self.curve_steps),
                    nn.Sigmoid(),
                    nn.Unflatten(1, (1, self.curve_steps, 1)),
                    Multiply(3.0),
                )
            elif param == "color":
                head = nn.Sequential(
                    nn.Linear(fc_dim, self.curve_steps * 3),
                    nn.Sigmoid(),
                    nn.Unflatten(1, (3, self.curve_steps, 1)),
                    Multiply(3.0),
                )
            elif param == "blur":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(10.0))

            if head is not None:
                param_heads[param] = head

        return param_heads

    def normalize_for_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize an image batch before MobileNet feature extraction.

        Pretrained timm models expect their training-time normalization. When the
        backbone is not pretrained, this function leaves the image unchanged.

        Args:
            x:
                Image tensor in [0, 1], shape [B, C, H, W].

        Returns:
            Normalized image tensor with the same shape as `x`.
        """
        if not self.mobilenet_pretrained:
            return x
        return (x - self.backbone_mean) / self.backbone_std

    def get_img_enc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch with the MobileNet backbone.

        The image is resized to the backbone's expected input resolution,
        normalized if pretrained weights are used, passed through
        `backbone.forward_features`, and globally averaged over the spatial
        dimensions.

        Args:
            x:
                Input image tensor in [0, 1], shape [B, C, H, W].

        Returns:
            Image feature tensor of shape [B, encoder_dim].
        """
        x_enc = normalize_for_mobilenet(x, self.backbone_input_hw)
        x_enc = self.normalize_for_backbone(x_enc)
        feat: torch.Tensor = self.backbone.forward_features(x_enc)
        return feat.mean(dim=[2, 3])

    def get_predicted_params(
        self,
        img_features: torch.Tensor,
        alpha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict absolute transformation parameters from image features and alpha.

        Alpha is encoded into a condition vector and concatenated with the image
        feature vector. The shared MLP trunk produces a latent representation,
        which is then passed through one head per transformation parameter.

        Args:
            img_features:
                Encoded image features, shape [B, encoder_dim].

            alpha:
                Scalar condition tensor, shape [B] or [B, 1].

        Returns:
            Dictionary mapping parameter names to tensors accepted by
            `apply_params`.
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        cond = self.cond_encoder(alpha)
        x = torch.cat([img_features, cond], dim=1)

        expected_in_features = self.fc_layers[0].in_features
        assert x.shape[1] == expected_in_features, (
            f"Feature mismatch: got {x.shape[1]} features, "
            f"but dense0 expects {expected_in_features}"
        )
        # Apply the configurable fully connected trunk.
        for layer in self.fc_layers:
            x = F.leaky_relu(layer(x))

        dict_params = {}

        for param, head in self.param_heads.items():
            out: torch.Tensor = head(x)
            # Scalar heads return [B, 1]. Squeeze to [B] to preserve the
            # original output contract expected by apply_params.
            #
            # Curve heads are intentionally not squeezed because they have
            # structured shapes such as [B, 1, CURVE_STEPS, 1].
            if out.ndim == 2 and out.shape[1] == 1:
                out = out.squeeze(1)

            dict_params[param] = out

        return dict_params

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full MobileNet generator forward pass.

        Args:
            x:
                Input image tensor in [0, 1], shape [B, C, H, W].

            alpha:
                Scalar condition tensor, shape [B] or [B, 1].

        Returns:
            A tuple containing:

            final_img:
                Final transformed RGB image, shape [B, 3, H, W].

            flat_params:
                Flattened parameter tensor in the standard parameter order.
        """
        enc = self.get_img_enc(x)
        params = self.get_predicted_params(enc, alpha)

        self.img_list = apply_params(x[:, : self.color_channels, :, :], params)
        flat_params = flatten_output_params(params, x.shape[0])

        return self.img_list[-1], flat_params


class ConditionalParametricResidualMobileNetGenerator(nn.Module):
    """
    Best-performing conditional parametric generator architecture.

    This model uses a timm MobileNet backbone for image feature extraction and a
    residual, identity-aware parameterization for predicting image transformation
    parameters.

    Unlike the absolute MobileNet baseline, this model does not directly predict
    final transform values. Instead, each parameter head predicts a raw residual
    value. The residual is combined with the conditioning value `alpha` and then
    mapped into a valid bounded parameter range around an explicit identity
    value.

    Main design goal:
        alpha = 0 should produce little or no edit.

    Identity values used by the residual mapping:
        exposure   -> 0.0
        saturation -> 1.0
        contrast   -> 1.0
        sharp      -> 1.0
        tone       -> 1.0
        color      -> 1.0
        blur       -> very small near-zero value

    The model supports two shared-trunk variants:

        use_film=False:
            Original concat pathway:
                concat(image_features, alpha_embedding) -> MLP -> parameter heads

            This path is kept for backwards compatibility with older checkpoints.

        use_film=True:
            FiLM residual pathway:
                image_features -> projection -> FiLM residual blocks -> parameter heads

            In this path, alpha modulates the image latent representation through
            feature-wise affine transformations.

    Expected inputs:
        x:
            Image tensor in [0, 1], shape [B, C, H, W].

        alpha:
            Scalar condition tensor, shape [B] or [B, 1].

    Returns:
        final_img:
            Final transformed RGB image, shape [B, 3, H, W].

        flat_params:
            Flattened transformation parameter tensor, useful for logging,
            debugging, or auxiliary supervision.
    """

    def __init__(
        self,
        fc_dim: int = 128,
        fc_depth: int = 2,
        in_channels: int = 3,
        mobilenet_variant: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        backbone_input_size: int | None = None,
        mobilenet_pretrained: bool = True,
        freeze_backbone: bool = True,
        cond_dim: int = 32,
        tone_head_depth: int = 1,
        tone_head_dim: int | None = None,
        color_head_depth: int = 1,
        color_head_dim: int | None = None,
        use_film: bool = False,
        film_hidden_dim: int | None = None,
        use_res_head: bool = False,
        use_alpha_head: bool = True,
        use_sigmoid_head: bool = False,
    ) -> None:
        """
        Initialize the residual MobileNet generator.

        Args:
            fc_dim:
                Width of the shared latent representation and parameter-head
                input dimension.

            fc_depth:
                Number of shared trunk layers. In concat mode, this is the
                number of MLP layers. In FiLM mode, this is the number of FiLM
                residual blocks.

            in_channels:
                Number of image channels expected by the MobileNet backbone.

            mobilenet_variant:
                Name of the timm MobileNet variant to use.

            backbone_input_size:
                Optional manual square input resolution for the backbone. If
                None, the input size from the timm data config is used.

            mobilenet_pretrained:
                If True, load pretrained timm weights and normalize inputs with
                the timm mean/std.

            freeze_backbone:
                If True, keep the MobileNet backbone fixed as a feature
                extractor.

            cond_dim:
                Size of the learned alpha condition embedding.

            tone_head_depth:
                Number of layers in the tone-curve projection head. depth=1
                keeps the old checkpoint-compatible Linear head.

            tone_head_dim:
                Hidden dimension used by deeper tone heads. If None, uses
                fc_dim.

            color_head_depth:
                Number of layers in the color-curve projection head. depth=1
                keeps the old checkpoint-compatible Linear head.

            color_head_dim:
                Hidden dimension used by deeper color heads. If None, uses
                fc_dim.

            use_film:
                If False, use the original concat+MLP trunk. If True, use a FiLM
                residual trunk conditioned by alpha.

            film_hidden_dim:
                Hidden dimension used inside FiLM modulation layers. If None,
                uses fc_dim.

            use_res_head:
                If True, use the older multiplicative residual formula in
                compute_u:
                    alpha * (1 + gain * tanh(raw))

                If False, use the simpler alpha-scaled raw residual:
                    alpha * raw * gain

            use_alpha_head:
                If True, alpha controls edit strength. If False, alpha is
                ignored inside compute_u and treated as 1.

            use_sigmoid_head:
                If True, use shifted sigmoid mapping in transform_param. If
                False, use the original piecewise tanh mapping.
        """
        super().__init__()

        params = PARAMS

        if fc_dim < 1:
            raise ValueError(f"fc_dim must be >= 1, got {fc_dim}")
        if fc_depth < 1:
            raise ValueError(f"fc_depth must be >= 1, got {fc_depth}")
        if cond_dim < 1:
            raise ValueError(f"cond_dim must be >= 1, got {cond_dim}")

        self.fc_dim = fc_dim
        self.fc_depth = fc_depth
        self.cond_dim = cond_dim
        self.curve_steps = CURVE_STEPS
        self.color_channels = COLOR_CHANNELS
        self.mobilenet_pretrained = mobilenet_pretrained
        self.freeze_backbone = freeze_backbone
        self.tone_head_depth = tone_head_depth
        self.tone_head_dim = tone_head_dim if tone_head_dim is not None else fc_dim
        self.color_head_depth = color_head_depth
        self.color_head_dim = color_head_dim if color_head_dim is not None else fc_dim
        self.use_res_head = use_res_head
        self.use_alpha_head = use_alpha_head
        self.use_sigmoid_head = use_sigmoid_head

        # Stores intermediate images returned by apply_params.
        # The final generated image is self.img_list[-1].
        self.img_list = None

        self.use_film = use_film
        self.film_hidden_dim = (
            film_hidden_dim if film_hidden_dim is not None else fc_dim
        )

        # MobileNet backbone setup
        # The backbone is created with num_classes=0 and global_pool="" inside
        # create_mobilenet_backbone, so it returns spatial feature maps instead
        # of classification logits.
        self.backbone, data_cfg = create_mobilenet_backbone(
            mobilenet_variant=mobilenet_variant,
            in_channels=in_channels,
            mobilenet_pretrained=mobilenet_pretrained,
        )
        # Resolve preprocessing from timm. Pretrained backbones use the timm
        # mean/std; non-pretrained backbones use identity normalization.
        backbone_mean, backbone_std, self.backbone_input_hw = (
            resolve_backbone_stats_and_input_size(
                data_cfg=data_cfg,
                in_channels=in_channels,
                mobilenet_pretrained=mobilenet_pretrained,
                backbone_input_size=backbone_input_size,
            )
        )
        # Convenience/debugging attribute for square-input backbones.
        self.backbone_input_size = self.backbone_input_hw[0]

        freeze_mobilenet_backbone(self.backbone, freeze_backbone)
        # Infer feature dimension dynamically because different timm MobileNet
        # variants can expose different output channel counts.
        encoder_dim = infer_mobilenet_encoder_dim(
            backbone=self.backbone,
            in_channels=in_channels,
            backbone_input_hw=self.backbone_input_hw,
            backbone_mean=backbone_mean,
            backbone_std=backbone_std,
            mobilenet_pretrained=mobilenet_pretrained,
        )

        # Register normalization tensors as buffers so they move with model.to().
        # They are not trainable parameters.
        self.register_buffer(
            "backbone_mean",
            torch.tensor(backbone_mean, dtype=torch.float32).view(
                1, len(backbone_mean), 1, 1
            ),
        )
        self.register_buffer(
            "backbone_std",
            torch.tensor(backbone_std, dtype=torch.float32).view(
                1, len(backbone_std), 1, 1
            ),
        )

        self.cond_encoder = build_cond_encoder(cond_dim)

        # -
        # Shared latent trunk
        # Two alternatives are supported:
        #
        # 1. use_film=False:
        #    Old concat+MLP path. This preserves old state_dict structure.
        #
        # 2. use_film=True:
        #    FiLM path. Image features are projected first, then alpha modulates
        #    them through FiLM residual blocks.
        # -
        if self.use_film:
            self.input_proj = nn.Linear(encoder_dim, fc_dim)
            self.film_blocks = nn.ModuleList(
                [
                    FiLMResidualBlock(
                        feat_dim=fc_dim,
                        cond_dim=cond_dim,
                        hidden_dim=self.film_hidden_dim,
                    )
                    for _ in range(fc_depth)
                ]
            )

            # Keep the attribute present for easier inspection, but do not
            # register old-path layers in this branch.
            self.fc_layers = None
        else:
            fc_layers = []
            in_dim = encoder_dim + cond_dim
            for _ in range(fc_depth):
                fc_layers.append(nn.Linear(in_dim, fc_dim))
                in_dim = fc_dim

            self.fc_layers = nn.ModuleList(fc_layers)

            # Keep the attributes present for easier inspection.
            self.input_proj = None
            self.film_blocks = None

        self.param_heads = self.define_parameter_prediction_heads(fc_dim, params)

    def _make_curve_projection(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> nn.Module:
        """
        Build the projection used by tone and color curve heads.

        This helper is intentionally backward-compatible:

            depth=1:
                Returns a plain nn.Linear. This preserves old state_dict keys
                such as:
                    param_heads.tone.0.weight
                    param_heads.tone.0.bias

            depth>1:
                Returns a deeper MLP ending in a Linear projection.

        Args:
            in_dim:
                Input feature dimension.

            out_dim:
                Output feature dimension.

            hidden_dim:
                Hidden layer width for deeper curve heads.

            depth:
                Number of Linear layers in the projection.

        Returns:
            Projection module mapping [B, in_dim] to [B, out_dim].
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        if depth == 1:
            return nn.Linear(in_dim, out_dim)

        layers = []
        current_dim = in_dim

        for _ in range(depth - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    def define_parameter_prediction_heads(
        self,
        fc_dim: int,
        params: list[str] | tuple[str, ...],
    ) -> nn.ModuleDict:
        """
        Build residual parameter prediction heads.

        These heads do not directly output final transform values. They output
        raw residual values, which are later combined with alpha and mapped into
        valid transformation ranges by get_predicted_params.

        Scalar heads:
            sharp, exposure, contrast, saturation, blur
            produce shape [B, 1], later squeezed to [B].

        Curve heads:
            tone  -> [B, 1, CURVE_STEPS, 1]
            color -> [B, 3, CURVE_STEPS, 1]

        Args:
            fc_dim:
                Input dimension for each parameter head.

            params:
                Iterable of parameter names to build heads for.

        Returns:
            ModuleDict mapping parameter names to raw residual heads.
        """
        param_heads = nn.ModuleDict()

        for param in params:
            if param in {"sharp", "exposure", "contrast", "saturation", "blur"}:
                # Raw scalar residual. No activation here: bounding happens in
                # transform_param or transform_blur_param.
                param_heads[param] = nn.Linear(fc_dim, 1)

            elif param == "tone":
                # Raw tone residual curve.
                param_heads[param] = nn.Sequential(
                    self._make_curve_projection(
                        in_dim=fc_dim,
                        out_dim=self.curve_steps,
                        hidden_dim=self.tone_head_dim,
                        depth=self.tone_head_depth,
                    ),
                    nn.Unflatten(1, (1, self.curve_steps, 1)),
                )

            elif param == "color":
                # Raw per-channel color residual curves.
                param_heads[param] = nn.Sequential(
                    self._make_curve_projection(
                        in_dim=fc_dim,
                        out_dim=self.curve_steps * 3,
                        hidden_dim=self.color_head_dim,
                        depth=self.color_head_depth,
                    ),
                    nn.Unflatten(1, (3, self.curve_steps, 1)),
                )

        return param_heads

    def normalize_for_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize an image batch before MobileNet feature extraction.

        Pretrained timm backbones expect their training-time normalization. When
        the backbone is not pretrained, the image is left unchanged.

        Args:
            x:
                Image tensor in [0, 1], shape [B, C, H, W].

        Returns:
            Normalized image tensor with the same shape as x.
        """
        if not self.mobilenet_pretrained:
            return x
        return (x - self.backbone_mean) / self.backbone_std

    def get_img_enc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch with the MobileNet backbone.

        The image is resized to the backbone's expected input resolution,
        normalized if pretrained weights are used, passed through
        backbone.forward_features, and globally averaged over height and width.

        Args:
            x:
                Input image tensor in [0, 1], shape [B, C, H, W].

        Returns:
            Image feature tensor of shape [B, encoder_dim].
        """
        x_enc = normalize_for_mobilenet(
            x,
            self.backbone_input_hw,
        )
        x_enc = self.normalize_for_backbone(x_enc)
        # forward_features returns a spatial map because global_pool="" was used.
        feat: torch.Tensor = self.backbone.forward_features(x_enc)
        return feat.mean(dim=[2, 3])  # Explicit global average pooling.

    def compute_u(
        self,
        alpha: torch.Tensor,
        raw: torch.Tensor,
        gain: float = 1.0,
    ) -> torch.Tensor:
        """
        Combine the condition alpha with a raw residual head output.

        This produces the latent edit value u, which is later mapped into the
        final parameter range.

        Two modes are supported:

            use_res_head=False:
                u = alpha * raw * gain

                This is the simpler residual formulation. If alpha=0, then
                u=0, so transform_param returns the identity value.

            use_res_head=True:
                u = alpha * (1 + gain * tanh(raw))

                Older multiplicative formulation. Kept for experiment/backward
                compatibility.

        If use_alpha_head=False, alpha is ignored and replaced by 1.

        Args:
            alpha:
                Conditioning tensor, broadcastable to raw.

            raw:
                Raw output from a parameter head.

            gain:
                Optional multiplier controlling residual strength.

        Returns:
            Latent edit tensor u.
        """
        if not self.use_alpha_head:
            alpha = 1

        if self.use_res_head:
            return alpha * (1.0 + gain * torch.tanh(raw))  # old definition
        else:
            return alpha * raw * gain

    def transform_param(
        self,
        u: torch.Tensor,
        min_val: float,
        max_val: float,
        identity: float,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Map a latent edit value to a bounded transform parameter.

        This function chooses between two bounded mappings:

            use_sigmoid_head=False:
                piecewise tanh mapping

            use_sigmoid_head=True:
                shifted sigmoid mapping

        Both mappings preserve the identity point:

            u = 0 -> identity

        Args:
            u:
                Latent edit tensor.

            min_val:
                Lower bound of the transform parameter.

            max_val:
                Upper bound of the transform parameter.

            identity:
                Neutral value of the transform parameter.

            scale:
                Controls saturation speed for the piecewise tanh mapping.

        Returns:
            Bounded transform parameter tensor with the same shape as u.
        """
        validate_mapping_bounds(
            min_val=min_val,
            max_val=max_val,
            identity=identity,
        )

        if self.use_sigmoid_head:
            return shifted_sigmoid_mapping(
                u=u,
                min_val=min_val,
                max_val=max_val,
                identity=identity,
            )

        return piecewise_tanh_mapping(
            u=u,
            min_val=min_val,
            max_val=max_val,
            identity=identity,
            scale=scale,
        )

    def transform_blur_param(
        self,
        u: torch.Tensor,
        identity: float = 1e-4,
        max_val: float = 3.0,
        scale: float = 1.0,
        min_val: float = 1e-6,
    ) -> torch.Tensor:
        """
        Map a latent edit value to a non-negative blur parameter.

        Blur is handled separately because blur strength should not become
        negative and because positive and negative raw residuals should both be
        able to increase blur magnitude. Squaring u makes the mapping symmetric
        around zero:

            u = 0      -> identity
            large |u|  -> max_val

        Args:
            u:
                Latent blur edit tensor.

            identity:
                Near-neutral blur value.

            max_val:
                Maximum blur strength.

            scale:
                Controls how quickly blur approaches max_val.

            min_val:
                Numerical lower clamp.

        Returns:
            Bounded non-negative blur parameter tensor.
        """
        mag = (u / scale) ** 2
        out = identity + (max_val - identity) * (mag / (1.0 + mag))
        return torch.clamp(out, min=min_val, max=max_val)

    def _get_shared_latent(
        self,
        img_features: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the shared latent representation used by all parameter heads.

        Args:
            img_features:
                Encoded image features, shape [B, encoder_dim].

            alpha:
                Scalar condition tensor, shape [B] or [B, 1].

        Returns:
            A tuple containing:

            x:
                Shared latent tensor, shape [B, fc_dim].

            alpha:
                Alpha reshaped to [B, 1].
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        cond = self.cond_encoder(alpha)

        if self.use_film:
            assert self.input_proj is not None
            assert self.film_blocks is not None
            expected_in_features = self.input_proj.in_features
            assert img_features.shape[1] == expected_in_features, (
                f"Feature mismatch: got {img_features.shape[1]} features, "
                f"but input_proj expects {expected_in_features}"
            )

            # FiLM path:
            # Image features are projected first. The condition embedding then
            # modulates the latent through several residual FiLM blocks.
            x = F.leaky_relu(self.input_proj(img_features))
            for block in self.film_blocks:
                x = block(x, cond)
        else:
            assert self.fc_layers is not None

            # Backward-compatible concat path:
            # This mirrors the older architecture and keeps old checkpoints
            # loadable when use_film=False.
            x = torch.cat([img_features, cond], dim=1)

            expected_in_features = self.fc_layers[0].in_features
            assert x.shape[1] == expected_in_features, (
                f"Feature mismatch: got {x.shape[1]} features, "
                f"but dense0 expects {expected_in_features}"
            )

            for layer in self.fc_layers:
                x = F.leaky_relu(layer(x))

        return x, alpha

    def get_predicted_params(
        self,
        img_features: torch.Tensor,
        alpha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict and convert transformation parameters.

        The parameter heads first produce raw residuals. These raw values are
        then combined with alpha and mapped into bounded parameter ranges around
        identity values.

        Args:
            img_features:
                Encoded image features, shape [B, encoder_dim].

            alpha:
                Scalar condition tensor, shape [B] or [B, 1].

        Returns:
            Dictionary mapping transformation names to valid parameter tensors
            accepted by apply_params.
        """
        x, alpha = self._get_shared_latent(img_features, alpha)
        # Scalar parameters use alpha as [B].
        alpha_scalar = alpha.squeeze(1)  # [B]
        # Curve parameters need alpha broadcastable to:
        # tone:  [B, 1, CURVE_STEPS, 1]
        # color: [B, 3, CURVE_STEPS, 1]
        alpha_curve = alpha.view(-1, 1, 1, 1)  # [B,1,1,1]

        dict_params = {}

        for param, head in self.param_heads.items():
            raw: torch.Tensor = head(x)

            if raw.ndim == 2 and raw.shape[1] == 1:
                raw = raw.squeeze(1)

            if param == "exposure":
                u = self.compute_u(alpha_scalar, raw)
                dict_params[param] = self.transform_param(
                    u,
                    min_val=-0.1,
                    max_val=1.0,
                    identity=0.0,
                    scale=1.0,
                )
            elif param == "saturation":
                u = self.compute_u(alpha_scalar, raw)
                dict_params[param] = self.transform_param(
                    u,
                    min_val=0,
                    max_val=3,
                    identity=1.0,
                    scale=1.0,
                )
            elif param == "contrast":
                u = self.compute_u(alpha_scalar, raw)
                dict_params[param] = self.transform_param(
                    u,
                    min_val=0.5,
                    max_val=3,
                    identity=1.0,
                    scale=1.0,
                )
            elif param == "sharp":
                u = self.compute_u(alpha_scalar, raw)
                dict_params[param] = self.transform_param(
                    u,
                    min_val=0.5,
                    max_val=3,
                    identity=1.0,
                    scale=1.0,
                )
            elif param == "blur":
                u = self.compute_u(alpha_scalar, raw)
                dict_params[param] = self.transform_blur_param(u)
            elif param == "tone":
                u = self.compute_u(alpha_curve, raw)
                dict_params[param] = self.transform_param(
                    u,
                    min_val=0.0,
                    max_val=3,
                    identity=1.0,
                    scale=1.0,
                )
            elif param == "color":
                u = self.compute_u(alpha_curve, raw)
                dict_params[param] = self.transform_param(
                    u,
                    min_val=0.0,
                    max_val=3,
                    identity=1.0,
                    scale=1.0,
                )
            else:
                # Fallback for any future parameter head that does not yet have
                # an explicit residual-to-parameter conversion.
                dict_params[param] = raw

        return dict_params

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full residual MobileNet generator forward pass.

        Args:
            x:
                Input image tensor in [0, 1], shape [B, C, H, W].

            alpha:
                Scalar condition tensor, shape [B] or [B, 1].

        Returns:
            A tuple containing:

            final_img:
                Final transformed RGB image, shape [B, 3, H, W].

            flat_params:
                Flattened parameter tensor in the standard parameter order.
        """
        enc = self.get_img_enc(x)
        params = self.get_predicted_params(enc, alpha)

        self.img_list = apply_params(x[:, : self.color_channels, :, :], params)
        flat_params = flatten_output_params(params, x.shape[0])

        return self.img_list[-1], flat_params

    def train(
        self, mode: bool = True
    ) -> "ConditionalParametricResidualMobileNetGenerator":
        """
        Set training/eval mode while respecting frozen-backbone behavior.

        PyTorch's default train(mode) would put all child modules into training
        mode. If the MobileNet backbone is frozen, this override keeps the
        backbone in eval mode even when the rest of the generator is trained.

        Args:
            mode:
                True for train mode, False for eval mode.

        Returns:
            self
        """
        super().train(mode)

        if self.freeze_backbone:
            self.backbone.eval()
        else:
            self.backbone.train(mode)

        return self
