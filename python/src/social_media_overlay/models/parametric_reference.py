"""
Legacy conditional parametric image generator.
It is kept as reference.

This file contains the earliest, monolithic implementation of the conditional
parametric generator. It is kept mainly for  reference and for loading
or comparing older experiments/checkpoints. New code should generally use the
refactored generator classes instead, for example:

    ConditionalParametricConvGenerator
    ConditionalParametricMobileNetGenerator
    ConditionalParametricResidualMobileNetGenerator
    MultiConditionalParametricResidualMobileNetGenerator

The original ConditionalParametricGenerator combines several responsibilities in
one class:

    1. It builds either a custom convolutional encoder or a timm MobileNetV4
       backbone, selected through the `backbone_type` argument.

    2. It resolves timm preprocessing, stores backbone mean/std buffers, resizes
       images for MobileNet, optionally freezes the backbone, and infers the
       MobileNet encoder dimension with a dummy forward pass.

    3. It encodes the scalar condition `alpha` with a small MLP and concatenates
       this condition embedding with the image features.

    4. It passes the concatenated feature vector through a fixed two-layer MLP
       (`dense0`, `dense1`) and predicts transformation parameters with one head
       per image operation.

    5. It applies the predicted parameters through `apply_params` and returns the
       final transformed image together with a flattened parameter tensor.

The old class uses an absolute parameterization. Its heads directly predict the
actual values passed to `apply_params`. For example, saturation, contrast,
sharpness, tone, color, and blur are produced with sigmoid-based heads scaled
into positive ranges, while exposure uses tanh scaled around zero. This means
`alpha` influences the output only because it is concatenated into the network
input; the final parameter values are not explicitly anchored to identity when
`alpha = 0`.

By contrast, the residual refactored models predict deltas or latent values
around identity and then convert them into valid transformation parameters.
Those versions are designed so that `alpha = 0` should stay close to an identity
mapping. They also use bounded conversion functions such as `transform_param`
and `transform_blur_param` instead of relying only on the output activation of
each prediction head. This makes the behavior safer and more interpretable,
especially when the model should learn the strength or direction of an edit
relative to a neutral image transform.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from social_media_overlay.ext.gebhardt.image_transformations.image_transformations import (
    apply_params,
)

import timm
from timm.data import resolve_model_data_config

from social_media_overlay.models.parametric_utils import PARAMS


class ConditionalParametricGenerator(nn.Module):
    def __init__(
        self,
        num_features,
        fc_dim,
        num_conv_layers=7,
        params=None,
        do_norm=True,
        in_channels=3,
        backbone_type="mobilenetv4",
        mobilenet_variant="mobilenetv4_conv_small.e2400_r224_in1k",
        backbone_input_size=None,  # optional manual override, otherwise infer from timm
        mobilenet_pretrained=True,
        freeze_backbone=True,
        cond_dim=32,
    ):
        super().__init__()

        self.num_features = num_features
        self.do_norm = do_norm
        self.fc_dim = fc_dim
        self.curve_steps = 8
        self.color_channels = 3
        self.cond_dim = cond_dim

        self.backbone_type = backbone_type
        self.mobilenet_pretrained = mobilenet_pretrained

        self.img_list = None

        if params is None:
            params = PARAMS

        # defaults for non-pretrained / non-timm cases
        backbone_mean = (0.0,) * in_channels
        backbone_std = (1.0,) * in_channels
        self.backbone_input_hw = None

        if backbone_type == "conv":
            self.conv_layers = nn.ModuleList()
            self.backbone = None

            out_channels_factor = 1
            cur_in_channels = in_channels

            for i in range(num_conv_layers):
                if i == 0:
                    padding = 3
                    kernel_size = 7
                else:
                    padding = 1
                    kernel_size = 3

                out_channels = self.num_features * out_channels_factor
                self.conv_layers.append(
                    self.general_conv2d(
                        in_channels=cur_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        do_norm=do_norm,
                    )
                )
                cur_in_channels = out_channels
                out_channels_factor *= 2

            out_channels_factor = out_channels_factor // 2
            encoder_dim = self.num_features * out_channels_factor

            # not used for conv backbone, but keep a sensible value
            if backbone_input_size is not None:
                self.backbone_input_hw = (backbone_input_size, backbone_input_size)

        elif backbone_type == "mobilenetv4":
            self.backbone = timm.create_model(
                mobilenet_variant,
                pretrained=mobilenet_pretrained,
                in_chans=in_channels,
                num_classes=0,
                global_pool="",
            )
            self.conv_layers = None

            # fetch model-specific preprocessing from timm
            data_cfg = resolve_model_data_config(self.backbone)

            if mobilenet_pretrained:
                backbone_mean = tuple(data_cfg["mean"])
                backbone_std = tuple(data_cfg["std"])

            # input_size is (C, H, W)
            _, cfg_h, cfg_w = data_cfg["input_size"]

            if backbone_input_size is None:
                self.backbone_input_hw = (cfg_h, cfg_w)
            else:
                # manual override if user wants one fixed square size
                self.backbone_input_hw = (backbone_input_size, backbone_input_size)

            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            # freeze unused final head params
            self.backbone.conv_head.weight.requires_grad = False
            self.backbone.norm_head.weight.requires_grad = False
            self.backbone.norm_head.bias.requires_grad = False

            # infer encoder feature dimension using the resolved backbone input size
            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(
                    1, in_channels, self.backbone_input_hw[0], self.backbone_input_hw[1]
                )

                if mobilenet_pretrained:
                    mean = torch.tensor(backbone_mean, dtype=dummy.dtype).view(
                        1, in_channels, 1, 1
                    )
                    std = torch.tensor(backbone_std, dtype=dummy.dtype).view(
                        1, in_channels, 1, 1
                    )
                    dummy = (dummy - mean) / std

                feat = self.backbone.forward_features(dummy)
                encoder_dim = feat.shape[1]

            self.backbone.train(was_training)

        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # keep old attribute name too, for convenience / debugging
        self.backbone_input_size = (
            None if self.backbone_input_hw is None else self.backbone_input_hw[0]
        )

        # store normalization tensors as buffers so they move with .to(device)
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

        self.cond_encoder = nn.Sequential(
            nn.Linear(1, self.cond_dim),
            nn.LeakyReLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.LeakyReLU(),
        )

        self.dense0 = nn.Linear(encoder_dim + self.cond_dim, fc_dim)
        self.dense1 = nn.Linear(fc_dim, fc_dim)

        self.param_heads = self.define_parameter_prediction_heads(fc_dim, params)

    def normalize_for_backbone(self, x):
        """
        Normalize image tensor for pretrained timm backbone.
        Expects x in [0,1].
        """
        if not self.mobilenet_pretrained:
            return x
        return (x - self.backbone_mean) / self.backbone_std

    def define_parameter_prediction_heads(self, fc_dim, params):
        param_heads = nn.ModuleDict()

        for param in params:
            if param == "affine":
                head_trans = nn.Sequential(
                    nn.Linear(fc_dim, 2),
                    nn.Tanh(),
                    nn.Unflatten(1, (2, 1)),
                    Multiply(100.0),
                )
                head_rot = nn.Sequential(
                    nn.Linear(fc_dim, 4),
                    nn.Tanh(),
                    nn.Unflatten(1, (2, 2)),
                    Multiply(0.25),
                )
                param_heads["trans"] = head_trans
                param_heads["rot"] = head_rot
                continue

            head = None

            if param == "gamma":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(2.0))
            elif param == "sharp":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(15.0))
            elif param == "wb":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(10.0))
            elif param == "exposure":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(3.0))
            elif param == "bright":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid())
            elif param == "contrast":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(3.0))
            elif param == "saturation":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid(), Multiply(10.0))
            elif param == "bw":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Sigmoid())
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
            elif param == "hue":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(math.pi))
            elif param == "scale":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 2),
                    nn.Sigmoid(),
                    nn.Unflatten(1, (2,)),
                    Multiply(3.0),
                )

            if head is not None:
                param_heads[param] = head

        return param_heads

    def forward(self, x, alpha):
        """
        Expects x in [0,1].
        """
        enc = self.get_img_enc(x)
        params = self.get_predicted_params(enc, alpha)

        self.img_list = apply_params(x[:, : self.color_channels, :, :], params)
        flat_params = torch.cat(
            [
                params["exposure"].reshape(x.shape[0], -1),  # [B, 1]
                params["saturation"].reshape(x.shape[0], -1),  # [B, 1]
                params["tone"].reshape(x.shape[0], -1),  # [B, 8]
                params["color"].reshape(x.shape[0], -1),  # [B, 24]
                params["contrast"].reshape(x.shape[0], -1),  # [B, 1]
                params["sharp"].reshape(x.shape[0], -1),  # [B, 1]
                params["blur"].reshape(x.shape[0], -1),  # [B, 1]
            ],
            dim=1,
        )
        return self.img_list[-1], flat_params

    def get_img_enc(self, x):
        """
        Encode image features.
        x is expected in [0,1].
        """
        if self.backbone_type == "mobilenetv4":
            x_enc = F.interpolate(
                x,
                size=self.backbone_input_hw,
                mode="bicubic",
                align_corners=False,  # not enabled in mobilenetv4
                antialias=False,  # ONNX doesn support interpolation with antialias
            )

            # normalize only for pretrained timm backbone
            x_enc = self.normalize_for_backbone(x_enc)

            feat = self.backbone.forward_features(x_enc)
            feat = feat.mean(dim=[2, 3])
            return feat

        for layer in self.conv_layers:
            x = layer(x)
        x = x.mean(dim=[2, 3])
        return x

    def get_predicted_params(self, x, alpha):
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        cond = self.cond_encoder(alpha)
        x = torch.cat([x, cond], dim=1)

        assert x.shape[1] == self.dense0.in_features, (
            f"Feature mismatch: got {x.shape[1]} features, "
            f"but dense0 expects {self.dense0.in_features}"
        )

        x = F.leaky_relu(self.dense0(x))
        x = F.leaky_relu(self.dense1(x))

        dict_params = {}
        for param, head in self.param_heads.items():
            if param in ("trans", "rot"):
                if "affine" in dict_params:
                    continue

                translation_param = self.param_heads["trans"](x)
                matrix_eye = (
                    torch.eye(2, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
                )
                rotation_param = self.param_heads["rot"](x) + matrix_eye
                dict_params["affine"] = torch.cat(
                    (rotation_param, translation_param), dim=2
                )
            else:
                out = head(x)

                if out.ndim == 2 and out.shape[1] == 1:
                    out = out.squeeze(1)

                dict_params[param] = out

        return dict_params

    @staticmethod
    def general_conv2d(
        in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding="valid",
        do_norm=True,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(out_channels) if do_norm else nn.Identity(),
        )


class ConditionalParametricResidualGenerator(nn.Module):
    """
    Generator that learns a parametric global transformation of images.

    Design:
    - input images are expected in [0,1]
    - parametric image transforms are applied directly in [0,1]
    - only the timm backbone input is normalized if needed
    - mean/std and input size are fetched from timm for pretrained backbones
    """

    def __init__(
        self,
        num_features,
        fc_dim,
        num_conv_layers=7,
        params=None,
        do_norm=True,
        in_channels=3,
        backbone_type="mobilenetv4",
        mobilenet_variant="mobilenetv4_conv_small.e2400_r224_in1k",
        backbone_input_size=None,
        mobilenet_pretrained=True,
        freeze_backbone=True,
        cond_dim=32,
    ):
        super().__init__()

        self.num_features = num_features
        self.do_norm = do_norm
        self.fc_dim = fc_dim
        self.curve_steps = 8
        self.color_channels = 3
        self.cond_dim = cond_dim

        self.backbone_type = backbone_type
        self.mobilenet_pretrained = mobilenet_pretrained

        self.img_list = None

        if params is None:
            params = [
                "exposure",
                "saturation",
                "tone",
                "color",
                "contrast",
                "sharp",
                "blur",
            ]

        backbone_mean = (0.0,) * in_channels
        backbone_std = (1.0,) * in_channels
        self.backbone_input_hw = None

        if backbone_type == "conv":
            self.conv_layers = nn.ModuleList()
            self.backbone = None

            out_channels_factor = 1
            cur_in_channels = in_channels

            for i in range(num_conv_layers):
                if i == 0:
                    padding = 3
                    kernel_size = 7
                else:
                    padding = 1
                    kernel_size = 3

                out_channels = self.num_features * out_channels_factor
                self.conv_layers.append(
                    self.general_conv2d(
                        in_channels=cur_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        do_norm=do_norm,
                    )
                )
                cur_in_channels = out_channels
                out_channels_factor *= 2

            out_channels_factor = out_channels_factor // 2
            encoder_dim = self.num_features * out_channels_factor

            if backbone_input_size is not None:
                self.backbone_input_hw = (backbone_input_size, backbone_input_size)

        elif backbone_type == "mobilenetv4":
            self.backbone = timm.create_model(
                mobilenet_variant,
                pretrained=mobilenet_pretrained,
                in_chans=in_channels,
                num_classes=0,
                global_pool="",
            )
            self.conv_layers = None

            data_cfg = resolve_model_data_config(self.backbone)

            if mobilenet_pretrained:
                backbone_mean = tuple(data_cfg["mean"])
                backbone_std = tuple(data_cfg["std"])

            _, cfg_h, cfg_w = data_cfg["input_size"]

            if backbone_input_size is None:
                self.backbone_input_hw = (cfg_h, cfg_w)
            else:
                self.backbone_input_hw = (backbone_input_size, backbone_input_size)

            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False

            self.backbone.conv_head.weight.requires_grad = False
            self.backbone.norm_head.weight.requires_grad = False
            self.backbone.norm_head.bias.requires_grad = False

            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(
                    1, in_channels, self.backbone_input_hw[0], self.backbone_input_hw[1]
                )

                if mobilenet_pretrained:
                    mean = torch.tensor(backbone_mean, dtype=dummy.dtype).view(
                        1, in_channels, 1, 1
                    )
                    std = torch.tensor(backbone_std, dtype=dummy.dtype).view(
                        1, in_channels, 1, 1
                    )
                    dummy = (dummy - mean) / std

                feat = self.backbone.forward_features(dummy)
                encoder_dim = feat.shape[1]

            self.backbone.train(was_training)

        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        self.backbone_input_size = (
            None if self.backbone_input_hw is None else self.backbone_input_hw[0]
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

        self.cond_encoder = nn.Sequential(
            nn.Linear(1, self.cond_dim),
            nn.LeakyReLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.LeakyReLU(),
        )

        self.dense0 = nn.Linear(encoder_dim + self.cond_dim, fc_dim)
        self.dense1 = nn.Linear(fc_dim, fc_dim)

        self.param_heads = self.define_parameter_prediction_heads(fc_dim, params)

    def normalize_for_backbone(self, x):
        if not self.mobilenet_pretrained:
            return x
        return (x - self.backbone_mean) / self.backbone_std

    def define_parameter_prediction_heads(self, fc_dim, params):
        param_heads = nn.ModuleDict()

        for param in params:
            if param == "affine":
                head_trans = nn.Sequential(
                    nn.Linear(fc_dim, 2),
                    nn.Tanh(),
                    nn.Unflatten(1, (2, 1)),
                    Multiply(0.25),
                )
                head_rot = nn.Sequential(
                    nn.Linear(fc_dim, 4),
                    nn.Tanh(),
                    nn.Unflatten(1, (2, 2)),
                    Multiply(0.25),
                )
                param_heads["trans"] = head_trans
                param_heads["rot"] = head_rot
                continue

            head = None

            # residual heads around 0, with safer scales
            if param == "gamma":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(1.0))
            elif param == "sharp":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(2.0)
                )  # was 3.0
            elif param == "wb":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(10.0))
            elif param == "exposure":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(2.0)
                )  # was 3.0
            elif param == "bright":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(1.0))
            elif param == "contrast":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(1.0)
                )  # was 2.0
            elif param == "saturation":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(2.0)
                )  # was 4.0
            elif param == "bw":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(1.0))
            elif param == "tone":
                head = nn.Sequential(
                    nn.Linear(fc_dim, self.curve_steps),
                    nn.Tanh(),
                    nn.Unflatten(1, (1, self.curve_steps, 1)),
                    Multiply(0.5),  # was 2.0
                )
            elif param == "color":
                head = nn.Sequential(
                    nn.Linear(fc_dim, self.curve_steps * 3),
                    nn.Tanh(),
                    nn.Unflatten(1, (3, self.curve_steps, 1)),
                    Multiply(0.5),  # was 2.0
                )
            elif param == "blur":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(0.5)
                )  # was 1.0
            elif param == "hue":
                head = nn.Sequential(nn.Linear(fc_dim, 1), nn.Tanh(), Multiply(math.pi))
            elif param == "scale":
                head = nn.Sequential(
                    nn.Linear(fc_dim, 2),
                    nn.Tanh(),
                    nn.Unflatten(1, (2,)),
                    Multiply(1.0),
                )

            if head is not None:
                param_heads[param] = head

        return param_heads

    # CHANGED: explicit identity builder using provided identities
    def get_identity_params(self, batch_size, device, dtype):
        identity = {
            "saturation": torch.full((batch_size,), 1.0, device=device, dtype=dtype),
            "exposure": torch.full((batch_size,), 0.0, device=device, dtype=dtype),
            "tone": torch.ones(
                batch_size, 1, self.curve_steps, 1, device=device, dtype=dtype
            ),
            # we expand identity to match the actual predicted color shape
            "color": torch.ones(
                batch_size, 3, self.curve_steps, 1, device=device, dtype=dtype
            ),
            "contrast": torch.full((batch_size,), 1.0, device=device, dtype=dtype),
            "sharp": torch.full((batch_size,), 1.0, device=device, dtype=dtype),
            "blur": torch.full((batch_size,), 0.1, device=device, dtype=dtype),
        }
        return identity

    def forward(self, x, alpha):
        enc = self.get_img_enc(x)
        params = self.get_predicted_params(enc, alpha)
        params = self.clamp_params(params)  # clamp into valid range!
        self.img_list = apply_params(x[:, : self.color_channels, :, :], params)

        flat_params = torch.cat(
            [
                params["exposure"].reshape(x.shape[0], -1),
                params["saturation"].reshape(x.shape[0], -1),
                params["tone"].reshape(x.shape[0], -1),
                params["color"].reshape(x.shape[0], -1),
                params["contrast"].reshape(x.shape[0], -1),
                params["sharp"].reshape(x.shape[0], -1),
                params["blur"].reshape(x.shape[0], -1),
            ],
            dim=1,
        )

        return self.img_list[-1], flat_params

    def clamp_params(self, params):
        params["exposure"] = params["exposure"].clamp(-3.0, 3.0)
        params["saturation"] = params["saturation"].clamp(max=10.0)
        params["contrast"] = params["contrast"].clamp(max=3.0)
        params["sharp"] = params["sharp"].clamp(max=15.0)
        params["blur"] = params["blur"].clamp(max=10.0)
        params["tone"] = params["tone"].clamp(max=3.0)
        params["color"] = params["color"].clamp(max=3.0)
        return params

    def get_img_enc(self, x):
        if self.backbone_type == "mobilenetv4":
            x_enc = F.interpolate(
                x,
                size=self.backbone_input_hw,
                mode="bicubic",
                align_corners=False,
                antialias=False,
            )

            x_enc = self.normalize_for_backbone(x_enc)

            feat = self.backbone.forward_features(x_enc)
            feat = feat.mean(dim=[2, 3])  # why this?
            return feat

        for layer in self.conv_layers:
            x = layer(x)
        x = x.mean(dim=[2, 3])
        return x

    def get_predicted_params(self, x, alpha):
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        cond = self.cond_encoder(alpha)
        x = torch.cat([x, cond], dim=1)

        x = F.leaky_relu(self.dense0(x))
        x = F.leaky_relu(self.dense1(x))

        batch_size = x.shape[0]
        dtype = x.dtype
        device = x.device

        dict_params = {}

        alpha_scalar = alpha.squeeze(1)  # [B]
        alpha_curve = alpha.view(-1, 1, 1, 1)  # [B,1,1,1]
        alpha_affine = alpha.view(-1, 1, 1)  # [B,1,1]

        for param, head in self.param_heads.items():
            if param in ("trans", "rot"):
                if "affine" in dict_params:
                    continue

                translation_delta = self.param_heads["trans"](x)
                rotation_delta = self.param_heads["rot"](x)

                matrix_eye = (
                    torch.eye(2, device=device, dtype=dtype)
                    .unsqueeze(0)
                    .repeat(batch_size, 1, 1)
                )

                rotation_param = matrix_eye + alpha_affine * rotation_delta
                translation_param = alpha_affine * translation_delta
                dict_params["affine"] = torch.cat(
                    (rotation_param, translation_param), dim=2
                )
                continue

            residual = head(x)

            if residual.ndim == 2 and residual.shape[1] == 1:
                residual = residual.squeeze(1)

            if param == "exposure":
                dict_params[param] = alpha_scalar * residual

            elif param == "saturation":
                dict_params[param] = torch.exp(alpha_scalar * residual)

            elif param == "contrast":
                dict_params[param] = torch.exp(alpha_scalar * residual)

            elif param == "sharp":
                dict_params[param] = torch.exp(alpha_scalar * residual)

            elif param == "blur":
                dict_params[param] = 0.1 * torch.exp(alpha_scalar * residual)

            elif param == "tone":
                dict_params[param] = torch.exp(alpha_curve * residual)

            elif param == "color":
                dict_params[param] = torch.exp(alpha_curve * residual)

            else:
                dict_params[param] = alpha_scalar * residual

        return dict_params

    @staticmethod
    def general_conv2d(
        in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding="valid",
        do_norm=True,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(out_channels) if do_norm else nn.Identity(),
        )


class Multiply(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * self.alpha
