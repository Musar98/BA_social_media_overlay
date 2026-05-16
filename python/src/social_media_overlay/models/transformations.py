import math
from typing import Mapping
import torch
import torch.nn as nn
from social_media_overlay.ext.gebhardt.image_transformations.image_transformations import (
    apply_params,
)
from social_media_overlay.models.parametric_utils import PARAMS, unflatten_output_params


class Transformations(nn.Module):
    """
    Testing model for applying differentiable image transformations.

    This module wraps the differentiable image transformation functions from
    Gebhardt et al. and applies a fixed, explicitly defined sequence of image
    transformations. The order of the transformations is important because each
    transformation is applied to the output of the previous one.

    The actual execution order is determined by the order of the parameter keys
    passed to `apply_params`. Since dictionaries preserve insertion order in
    modern Python versions, the order of `self.transformations` should match the
    order in which the corresponding parameter dictionary is later constructed.

    This model is mainly used for ONNX export and for comparing the performance
    of the PyTorch implementation against a fragment shader implementation.

    Args:
        transformations (Optional[list[str]], optional): Names of the image
            transformations to apply. If `None`, the default order is
            `["exposure", "saturation", "tone", "color", "contrast", "sharp", "blur"]`.
        in_channels (int, optional): Number of input image channels. Defaults to
            `3`.

    Attributes:
        transformations (list[str]): Ordered list of transformation names.
        color_channels (int): Number of image channels used by the input image.
    """

    def __init__(self, transformations=None, in_channels=3):
        super().__init__()
        if transformations is None:
            transformations = PARAMS

        self.transformations = transformations
        self.color_channels = in_channels

    def forward(self, x: torch.Tensor, params: torch.Tensor):
        """
        Apply the configured image transformation pipeline to the input image.

        The flat parameter tensor `params` is first converted into a dictionary using
        `unflatten_output_params`. The dictionary is then reordered according to
        `self.transformations`, clamped to valid transformation-specific ranges, and
        passed to Gebhardt et al.'s `apply_params` implementation.

        The effective execution order is defined by `self.transformations`, because
        the reordered parameter dictionary is built in that order and `apply_params`
        iterates over the dictionary keys.

        Only the first `self.color_channels` channels of `x` are transformed. This
        allows inputs with additional channels, such as masks or auxiliary maps, to
        keep only the image channels for the transformation pipeline.

        Args:
            x (Tensor): Input tensor with shape `(batch_size, channels, height, width)`.
                Only the first `self.color_channels` channels are passed to
                `apply_params`.
            params (Tensor): Flat tensor containing the transformation parameters.
                It is expected to match the structure required by
                `unflatten_output_params`.

        Returns:
            Tensor: Transformed image tensor with shape
            `(batch_size, self.color_channels, height, width)`.
        """
        params_dict = unflatten_output_params(params)
        params_dict = {
            k: params_dict[k] for k in self.transformations if k in params_dict
        }
        params_dict = self.clamp_params(params_dict)
        img_list = apply_params(x[:, : self.color_channels, :, :], params_dict)
        return img_list[-1]

    def clamp_params(self, params: Mapping[str, torch.Tensor]):
        params = params.copy()

        if "sharp" in params:
            params["sharp"] = params["sharp"].clamp(min=0.0)

        if "contrast" in params:
            params["contrast"] = params["contrast"].clamp(min=0.0)

        if "saturation" in params:
            params["saturation"] = params["saturation"].clamp(min=0.0)

        if "tone" in params:
            params["tone"] = params["tone"].clamp(min=0.0)

        if "color" in params:
            params["color"] = params["color"].clamp(min=0.0)

        if "blur" in params:
            params["blur"] = params["blur"].clamp(min=1e-6)

        return params
