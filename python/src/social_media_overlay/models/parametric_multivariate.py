import torch
import torch.nn as nn
import torch.nn.functional as F

from social_media_overlay.ext.gebhardt.image_transformations.image_transformations import (
    apply_params,
)
from social_media_overlay.models.film import FiLMResidualBlock
from social_media_overlay.models.parametric_utils import (
    COLOR_CHANNELS,
    CURVE_STEPS,
    PARAMS,
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


class MultiConditionalParametricResidualMobileNetGenerator(nn.Module):
    """
    Experimental multi-condition residual MobileNet generator using FiLM only.

    This class is a reference implementation for future work. It extends the
    residual MobileNet generator to a multivariate conditioning setting, but it
    has not been sufficiently evaluated or tested.

    Unlike the earlier multivariate prototype, this version does not use an
    extra condition-derived scalar, multiplicative residual modulation, or
    alpha_u-style control signal. The multi-condition input affects the model
    only through FiLM residual blocks.

    Architecture:

        image -> MobileNet backbone -> global average pooling -> image features
        condition vector -> condition encoder -> FiLM modulation
        image features -> projection -> FiLM residual blocks -> parameter heads
        raw parameter heads -> bounded identity-aware transform parameters

    Expected inputs:
        x:
            Image tensor in [0, 1], shape [B, C, H, W].

        alpha:
            Multi-condition tensor, shape [B, condition_dim].
            By default, condition_dim=2.

    Returns:
        final_img:
            Final transformed RGB image, shape [B, 3, H, W].

        flat_params:
            Flattened transformation parameter tensor.
    """

    def __init__(
        self,
        fc_dim: int = 128,
        fc_depth: int = 2,
        in_channels: int = 3,
        condition_dim: int = 2,
        mobilenet_variant: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        backbone_input_size: int | None = None,
        mobilenet_pretrained: bool = True,
        freeze_backbone: bool = True,
        cond_dim: int = 32,
        film_hidden_dim: int | None = None,
        tone_head_depth: int = 1,
        tone_head_dim: int | None = None,
        color_head_depth: int = 1,
        color_head_dim: int | None = None,
        use_sigmoid_head: bool = False,
    ) -> None:
        """
        Initialize the FiLM-only multi-condition residual MobileNet model.

        Args:
            fc_dim:
                Width of the shared latent representation.

            fc_depth:
                Number of FiLM residual blocks.

            in_channels:
                Number of image channels expected by the MobileNet backbone.

            condition_dim:
                Number of conditioning values per sample. The old experimental
                class used 2.

            mobilenet_variant:
                Name of the timm MobileNet variant to use.

            backbone_input_size:
                Optional manual square input resolution for the backbone.

            mobilenet_pretrained:
                If True, use pretrained timm weights and timm normalization.

            freeze_backbone:
                If True, keep MobileNet fixed as a feature extractor.

            cond_dim:
                Size of the encoded condition vector used by FiLM.

            film_hidden_dim:
                Hidden dimension inside FiLM modulation layers. If None, uses
                fc_dim.

            tone_head_depth:
                Number of Linear layers in the tone head projection.

            tone_head_dim:
                Hidden dimension for deeper tone heads. If None, uses fc_dim.

            color_head_depth:
                Number of Linear layers in the color head projection.

            color_head_dim:
                Hidden dimension for deeper color heads. If None, uses fc_dim.

            use_sigmoid_head:
                If True, use shifted sigmoid mapping in transform_param.
                If False, use piecewise tanh mapping.
        """
        super().__init__()

        params = PARAMS

        if fc_dim < 1:
            raise ValueError(f"fc_dim must be >= 1, got {fc_dim}")
        if fc_depth < 1:
            raise ValueError(f"fc_depth must be >= 1, got {fc_depth}")
        if cond_dim < 1:
            raise ValueError(f"cond_dim must be >= 1, got {cond_dim}")
        if condition_dim < 1:
            raise ValueError(f"condition_dim must be >= 1, got {condition_dim}")

        self.fc_dim = fc_dim
        self.fc_depth = fc_depth
        self.condition_dim = condition_dim
        self.cond_dim = cond_dim
        self.curve_steps = CURVE_STEPS
        self.color_channels = COLOR_CHANNELS
        self.mobilenet_pretrained = mobilenet_pretrained
        self.freeze_backbone = freeze_backbone
        self.use_sigmoid_head = use_sigmoid_head

        self.film_hidden_dim = (
            film_hidden_dim if film_hidden_dim is not None else fc_dim
        )

        self.tone_head_depth = tone_head_depth
        self.tone_head_dim = tone_head_dim if tone_head_dim is not None else fc_dim

        self.color_head_depth = color_head_depth
        self.color_head_dim = color_head_dim if color_head_dim is not None else fc_dim

        self.img_list = None

        # MobileNet backbone
        self.backbone, data_cfg = create_mobilenet_backbone(
            mobilenet_variant=mobilenet_variant,
            in_channels=in_channels,
            mobilenet_pretrained=mobilenet_pretrained,
        )

        backbone_mean, backbone_std, self.backbone_input_hw = (
            resolve_backbone_stats_and_input_size(
                data_cfg=data_cfg,
                in_channels=in_channels,
                mobilenet_pretrained=mobilenet_pretrained,
                backbone_input_size=backbone_input_size,
            )
        )

        self.backbone_input_size = self.backbone_input_hw[0]

        freeze_mobilenet_backbone(self.backbone, freeze_backbone)

        encoder_dim = infer_mobilenet_encoder_dim(
            backbone=self.backbone,
            in_channels=in_channels,
            backbone_input_hw=self.backbone_input_hw,
            backbone_mean=backbone_mean,
            backbone_std=backbone_std,
            mobilenet_pretrained=mobilenet_pretrained,
        )

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

        # FiLM-only multi-condition path
        # The condition vector is encoded once and then used only for FiLM
        # modulation. There is no cond_u, no alpha_u, and no multiplicative
        # residual control signal.
        self.cond_encoder = nn.Sequential(
            nn.Linear(condition_dim, cond_dim),
            nn.LeakyReLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LeakyReLU(),
        )

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

        self.param_heads = self.define_parameter_prediction_heads(fc_dim, params)

        # Optional but useful: initialize heads close to zero so the initial
        # bounded parameters start near identity.
        self._init_parameter_heads_near_identity()

    def _make_curve_projection(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> nn.Module:
        """
        Build the projection used by tone and color curve heads.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        if depth == 1:
            return nn.Linear(in_dim, out_dim)

        layers: list[nn.Module] = []
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
        Build raw residual parameter heads.

        The heads output unconstrained latent values. These values are converted
        into bounded image-transform parameters later.
        """
        param_heads = nn.ModuleDict()

        for param in params:
            if param in {"sharp", "exposure", "contrast", "saturation", "blur"}:
                param_heads[param] = nn.Linear(fc_dim, 1)

            elif param == "tone":
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

    def _init_parameter_heads_near_identity(self) -> None:
        """
        Initialize parameter heads near zero.

        Since transform_param maps u=0 to the identity value, zero-initialized
        final head layers make the initial model start close to no edit. This is
        especially helpful because the FiLM-only multivariate model does not use
        explicit alpha multiplication to force u=0.
        """
        for head in self.param_heads.values():
            modules = (
                list(head.modules()) if isinstance(head, nn.Sequential) else [head]
            )

            linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
            if not linear_layers:
                continue

            final_linear = linear_layers[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def normalize_for_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize an image batch before MobileNet feature extraction.
        """
        if not self.mobilenet_pretrained:
            return x

        mean = self.backbone_mean.to(dtype=x.dtype, device=x.device)
        std = self.backbone_std.to(dtype=x.dtype, device=x.device)

        return (x - mean) / std

    def get_img_enc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch with the MobileNet backbone.
        """
        x_enc = normalize_for_mobilenet(
            x,
            self.backbone_input_hw,
        )
        x_enc = self.normalize_for_backbone(x_enc)

        feat: torch.Tensor = self.backbone.forward_features(x_enc)

        return feat.mean(dim=[2, 3])

    def transform_param(
        self,
        u: torch.Tensor,
        min_val: float,
        max_val: float,
        identity: float,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Map a latent value to a bounded transform parameter.

        This preserves the identity point:

            u = 0 -> identity
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
        Map a latent value to a non-negative blur parameter.

        Blur is symmetric around zero because positive and negative latent
        values should both be able to increase blur strength.
        """
        mag = (u / scale) ** 2
        out = identity + (max_val - identity) * (mag / (1.0 + mag))

        return torch.clamp(out, min=min_val, max=max_val)

    def _get_shared_latent(
        self,
        img_features: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the FiLM-modulated latent representation.

        The condition is used only through FiLM. No condition-derived scalar
        multiplier is created.
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        if alpha.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected alpha with shape [B, {self.condition_dim}], "
                f"got {tuple(alpha.shape)}"
            )

        cond = self.cond_encoder(alpha)

        expected_in_features = self.input_proj.in_features
        assert img_features.shape[1] == expected_in_features, (
            f"Feature mismatch: got {img_features.shape[1]} features, "
            f"but input_proj expects {expected_in_features}"
        )

        x = F.leaky_relu(self.input_proj(img_features))

        for block in self.film_blocks:
            x = block(x, cond)

        return x

    def get_predicted_params(
        self,
        img_features: torch.Tensor,
        alpha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict transform parameters from image features and multivariate condition.

        The condition affects the shared latent only through FiLM blocks. The
        parameter heads output raw latent values directly. These raw values are
        then mapped into bounded transform ranges.
        """
        x = self._get_shared_latent(img_features, alpha)

        dict_params: dict[str, torch.Tensor] = {}

        for param, head in self.param_heads.items():
            raw: torch.Tensor = head(x)

            if raw.ndim == 2 and raw.shape[1] == 1:
                raw = raw.squeeze(1)

            if param == "exposure":
                dict_params[param] = self.transform_param(
                    raw,
                    min_val=-0.1,
                    max_val=1.0,
                    identity=0.0,
                    scale=1.0,
                )

            elif param == "saturation":
                dict_params[param] = self.transform_param(
                    raw,
                    min_val=0.0,
                    max_val=3.0,
                    identity=1.0,
                    scale=1.0,
                )

            elif param == "contrast":
                dict_params[param] = self.transform_param(
                    raw,
                    min_val=0.5,
                    max_val=3.0,
                    identity=1.0,
                    scale=1.0,
                )

            elif param == "sharp":
                dict_params[param] = self.transform_param(
                    raw,
                    min_val=0.5,
                    max_val=3.0,
                    identity=1.0,
                    scale=1.0,
                )

            elif param == "blur":
                dict_params[param] = self.transform_blur_param(raw)

            elif param == "tone":
                dict_params[param] = self.transform_param(
                    raw,
                    min_val=0.0,
                    max_val=3.0,
                    identity=1.0,
                    scale=1.0,
                )

            elif param == "color":
                dict_params[param] = self.transform_param(
                    raw,
                    min_val=0.0,
                    max_val=3.0,
                    identity=1.0,
                    scale=1.0,
                )

            else:
                dict_params[param] = raw

        return dict_params

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full FiLM-only multi-condition generator forward pass.
        """
        enc = self.get_img_enc(x)
        params = self.get_predicted_params(enc, alpha)

        self.img_list = apply_params(x[:, : self.color_channels, :, :], params)
        flat_params = flatten_output_params(params, x.shape[0])

        return self.img_list[-1], flat_params

    def train(
        self,
        mode: bool = True,
    ) -> "MultiConditionalParametricResidualMobileNetGenerator":
        """
        Set training/eval mode while respecting frozen-backbone behavior.
        """
        super().train(mode)

        if self.freeze_backbone:
            self.backbone.eval()
        else:
            self.backbone.train(mode)

        return self
