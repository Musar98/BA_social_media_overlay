from typing import Any, Mapping, Union

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from timm.data import resolve_model_data_config

# Number of control points used by the tone and color curve parameter heads.
# tone predicts shape [B, 1, CURVE_STEPS, 1]
# color predicts shape [B, 3, CURVE_STEPS, 1]
CURVE_STEPS = 8

# Number of image channels passed into the parametric color transformations.
# This is 3 for RGB. Extra channels, if present in x, are ignored by apply_params.
COLOR_CHANNELS = 3

"""
Default ordered list of image transformations.

The order of this list defines the intended execution order of the
transformation pipeline:

    1. exposure
    2. saturation
    3. tone
    4. color
    5. contrast
    6. sharp
    7. blur

This mirrors the ordering mechanism used in the Gebhardt et al. implementation:
a transformation-name list is passed to `init_params`, which inserts entries
into a parameter dictionary in that same order. The parameter dictionary is then
passed to `apply_params`, where transformations are applied by iterating over
the dictionary keys.

This relies on dictionary insertion order being preserved. That behavior is
guaranteed for standard dictionaries in Python 3.7 and later. In CPython 3.6 it
also works, but was considered an implementation detail. For older Python
versions, use `collections.OrderedDict` if this ordering must be guaranteed.
"""
PARAMS = ["exposure", "saturation", "tone", "color", "contrast", "sharp", "blur"]


def flatten_output_params(
    params: Mapping[str, torch.Tensor], batch_size: int
) -> torch.Tensor:
    """
    Flatten transformation parameters into a single batched parameter tensor.

    Parameters are flattened in the fixed order defined by `PARAMS`:

        1. exposure
        2. saturation
        3. tone
        4. color
        5. contrast
        6. sharp
        7. blur

    This order must match `unflatten_output_params`, because the flat tensor is
    later reconstructed by slicing fixed index ranges. Only keys that are
    present in `params` are included.

    Expected parameter shapes before flattening are typically:

        exposure:   `(batch_size,)` or `(batch_size, 1)`
        saturation: `(batch_size,)` or `(batch_size, 1)`
        tone:       `(batch_size, 1, 8, 1)`
        color:      `(batch_size, 3, 8, 1)`
        contrast:   `(batch_size,)` or `(batch_size, 1)`
        sharp:      `(batch_size,)` or `(batch_size, 1)`
        blur:       `(batch_size,)` or `(batch_size, 1)`

    Args:
        params (Mapping[str, Tensor]): Dictionary-like mapping from
            transformation names to parameter tensors.
        batch_size (int): Batch size used when reshaping each parameter tensor.

    Returns:
        Tensor: Flat tensor with shape `(batch_size, num_params)`. If no
        parameters are present, returns an empty tensor with shape
        `(batch_size, 0)`.
    """
    default_order = PARAMS
    # We did not use these following settings!
    # extra_order = {
    #    "gamma": None,
    #    "wb": None,
    #    "bright": None,
    #    "bw": None,
    #    "hue": None,
    #    "scale": None,
    #    "affine": None,
    # }

    flat = []

    # default params first
    for key in default_order:
        if key in params:
            flat.append(params[key].reshape(batch_size, -1))

    # extra params after
    # for key in extra_order:
    #    if key in params:
    #        flat.append(params[key].reshape(batch_size, -1))

    return torch.cat(flat, dim=1) if flat else torch.empty(batch_size, 0)


def unflatten_output_params(flat: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """
    Convert a flat batched parameter tensor back into a parameter dictionary.

    The expected full flat layout is:

        exposure:   `flat[:, 0]`
        saturation: `flat[:, 1]`
        tone:       `flat[:, 2:10]`
        color:      `flat[:, 10:34]`
        contrast:   `flat[:, 34]`
        sharp:      `flat[:, 35]`
        blur:       `flat[:, 36]`

    A full parameter tensor therefore has shape `(batch_size, 37)`. Partial
    tensors are supported by only reconstructing parameters whose full slice is
    available.

    Args:
        flat (Tensor): Flat parameter tensor with shape
            `(batch_size, num_params)`.

    Returns:
        dict[str, Tensor]: Dictionary mapping transformation names to
        reconstructed parameter tensors.
    """
    if flat.ndim != 2:
        raise ValueError(
            f"`flat` must be a 2D tensor, but got shape {tuple(flat.shape)}."
        )

    expected_num_params = 37
    if flat.shape[1] != expected_num_params:
        raise ValueError(
            f"`flat` must contain exactly {expected_num_params} parameters per sample, "
            f"but got {flat.shape[1]}."
        )

    params: dict[str, torch.Tensor] = {
        "exposure": flat[:, 0],
        "saturation": flat[:, 1],
        "tone": flat[:, 2:10].reshape(-1, 1, 8, 1),
        "color": flat[:, 10:34].reshape(-1, 3, 8, 1),
        "contrast": flat[:, 34],
        "sharp": flat[:, 35],
        "blur": flat[:, 36],
    }

    return params


def build_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple[int, int]] = 3,
    stride: Union[int, tuple[int, int]] = 1,
    padding: Union[str, int, tuple[int, int]] = "valid",
    do_norm: bool = True,
) -> nn.Sequential:
    """
    Build one convolutional encoder block.

    The block is used by the plain convolutional generator encoder and follows
    the same layout as the original implementation:

        Conv2d -> LeakyReLU -> optional InstanceNorm2d

    Args:
        in_channels:
            Number of input feature channels.

        out_channels:
            Number of output feature channels produced by the convolution.

        kernel_size:
            Convolution kernel size. Can be an integer or a pair of integers.

        stride:
            Convolution stride. In the encoder this is usually 2, so each block
            downsamples the spatial resolution.

        padding:
            Padding passed directly to nn.Conv2d. The encoder uses padding=3
            for the first 7x7 block and padding=1 for later 3x3 blocks.

        do_norm:
            If True, append InstanceNorm2d after the activation. If False, use
            Identity instead so the returned block has the same structure.

    Returns:
        A sequential PyTorch module containing the convolutional block.
    """
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


def build_conv_encoder(
    in_channels: int,
    num_features: int,
    num_conv_layers: int,
    do_norm: bool,
) -> tuple[nn.ModuleList, int]:
    """
    Build the convolutional image encoder used by the conv-only generator.

    The encoder follows the original Gebhardt-style architecture. It is a stack
    of strided convolutional blocks that progressively downsample the image while
    increasing the number of feature channels.

    The first block uses a larger 7x7 kernel to capture broader low-level image
    structure. All following blocks use 3x3 kernels. Every block uses stride=2,
    so each layer reduces the spatial resolution.

    Channel progression:

        layer 0: num_features
        layer 1: num_features * 2
        layer 2: num_features * 4
        ...

    Args:
        in_channels:
            Number of channels in the input image, usually 3 for RGB.

        num_features:
            Base number of feature channels used in the first convolutional
            block. The number of channels doubles after each block.

        num_conv_layers:
            Number of convolutional blocks to build.

        do_norm:
            If True, each block includes InstanceNorm2d after the activation.
            If False, the normalization layer is replaced with Identity.

    Returns:
        A tuple containing:

        conv_layers:
            ModuleList of convolutional blocks.

        encoder_dim:
            Number of channels produced by the final convolutional block. This
            is also the feature dimension after global average pooling.
    """
    conv_layers = nn.ModuleList()

    out_channels_factor = 1
    cur_in_channels = in_channels

    for i in range(num_conv_layers):
        if i == 0:
            kernel_size = 7
            padding = 3
        else:
            kernel_size = 3
            padding = 1

        out_channels = num_features * out_channels_factor
        conv_layers.append(
            build_conv_block(
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

    out_channels_factor //= 2
    encoder_dim = num_features * out_channels_factor
    return conv_layers, encoder_dim


def create_mobilenet_backbone(
    mobilenet_variant: str,
    in_channels: int,
    mobilenet_pretrained: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Create a timm MobileNet backbone for feature extraction.

    The model is configured as an image encoder, not as a classifier.
    Setting num_classes=0 removes the classification head, and setting
    global_pool="" keeps the spatial feature map instead of applying timm's
    final pooling internally. This lets the generator explicitly decide how to
    pool the features later, currently with global average pooling.

    Args:
        mobilenet_variant:
            Name of the timm MobileNet model variant to create, for example
            "mobilenetv4_conv_small.e2400_r224_in1k".

        in_channels:
            Number of input image channels. Usually 3 for RGB images.

        mobilenet_pretrained:
            If True, load pretrained weights from timm. If False, initialize the
            backbone randomly.

    Returns:
        A tuple containing:

        backbone:
            The timm MobileNet model configured to return spatial feature maps
            from `forward_features`.

        data_cfg:
            The timm data configuration for the backbone. This contains values
            such as input size, mean, and standard deviation, which are used for
            preprocessing.
    """
    backbone = timm.create_model(
        mobilenet_variant,
        pretrained=mobilenet_pretrained,
        in_chans=in_channels,
        num_classes=0,
        global_pool="",
    )
    data_cfg = resolve_model_data_config(backbone)
    return backbone, data_cfg


def resolve_backbone_stats_and_input_size(
    data_cfg: dict[str, Any],
    in_channels: int,
    mobilenet_pretrained: bool,
    backbone_input_size: int | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, int]]:
    """
    Resolve the normalization statistics and input resolution for a MobileNet backbone.

    Pretrained timm backbones expect the same preprocessing that was used during
    pretraining, so their mean, standard deviation, and input size are read from
    the timm data config. Non-pretrained backbones use identity normalization,
    meaning the input image is left in its original [0, 1] range.

    Args:
        data_cfg:
            Data configuration returned by timm for the selected backbone.
            Expected to contain keys such as "mean", "std", and "input_size".

        in_channels:
            Number of input image channels. Used to create identity
            normalization statistics when pretrained weights are not used.

        mobilenet_pretrained:
            If True, use the mean and standard deviation from `data_cfg`.
            If False, use mean 0 and standard deviation 1 for each input channel.

        backbone_input_size:
            Optional manual square input size for the backbone. If provided,
            the returned input size is `(backbone_input_size, backbone_input_size)`.
            If None, the height and width from `data_cfg["input_size"]` are used.

    Returns:
        A tuple containing:

        backbone_mean:
            Per-channel normalization mean used before the MobileNet backbone.

        backbone_std:
            Per-channel normalization standard deviation used before the
            MobileNet backbone.

        backbone_input_hw:
            Spatial input size as `(height, width)`. This is the size images are
            resized to before being passed through the backbone.
    """
    if mobilenet_pretrained:
        backbone_mean = tuple(data_cfg["mean"])
        backbone_std = tuple(data_cfg["std"])
    else:
        backbone_mean = (0.0,) * in_channels
        backbone_std = (1.0,) * in_channels

    _, cfg_h, cfg_w = data_cfg["input_size"]

    if backbone_input_size is None:
        backbone_input_hw = (cfg_h, cfg_w)
    else:
        backbone_input_hw = (backbone_input_size, backbone_input_size)

    return backbone_mean, backbone_std, backbone_input_hw


def freeze_mobilenet_backbone(
    backbone: nn.Module,
    freeze_backbone: bool,
) -> None:
    """
    Freeze MobileNet backbone parameters when requested.

    If `freeze_backbone` is True, all backbone parameters are marked as
    non-trainable and the backbone is switched to evaluation mode. This is used
    when MobileNet should act as a fixed feature extractor instead of being
    fine-tuned with the generator.

    The function also explicitly freezes known MobileNet head parameters when
    they exist. This preserves the behavior of the original implementation,
    where `conv_head` and `norm_head` were frozen manually. The attribute checks
    make the function safer across different timm MobileNet variants, because
    not every variant exposes exactly the same head modules.

    Args:
        backbone:
            The timm MobileNet model used as the image feature extractor.

        freeze_backbone:
            If True, freeze the full backbone and put it in eval mode. If False,
            leave the main backbone parameters trainable, but still freeze known
            unused head parameters when present.

    Returns:
        None. The function modifies `backbone` in place.
    """
    if freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

    if hasattr(backbone, "conv_head") and hasattr(backbone.conv_head, "weight"):
        backbone.conv_head.weight.requires_grad = False

    if hasattr(backbone, "norm_head"):
        if (
            hasattr(backbone.norm_head, "weight")
            and backbone.norm_head.weight is not None
        ):
            backbone.norm_head.weight.requires_grad = False
        if hasattr(backbone.norm_head, "bias") and backbone.norm_head.bias is not None:
            backbone.norm_head.bias.requires_grad = False


def infer_mobilenet_encoder_dim(
    backbone: nn.Module,
    in_channels: int,
    backbone_input_hw: tuple[int, int],
    backbone_mean: tuple[float, ...],
    backbone_std: tuple[float, ...],
    mobilenet_pretrained: bool,
) -> int:
    """
    Infer the feature dimension produced by a MobileNet backbone.

    Different timm MobileNet variants can return different numbers of feature
    channels. Instead of hardcoding this value, this function runs a dummy image
    through `backbone.forward_features` and reads the channel dimension from the
    resulting feature map.

    The backbone is temporarily switched to evaluation mode for the dummy
    forward pass and then restored to its previous training/eval state.

    Args:
        backbone:
            The timm MobileNet model used as the image feature extractor.

        in_channels:
            Number of channels expected by the backbone input.

        backbone_input_hw:
            Spatial input size as `(height, width)`. The dummy image is created
            with this resolution.

        backbone_mean:
            Per-channel normalization mean. Used only when `mobilenet_pretrained`
            is True.

        backbone_std:
            Per-channel normalization standard deviation. Used only when
            `mobilenet_pretrained` is True.

        mobilenet_pretrained:
            If True, normalize the dummy image with `backbone_mean` and
            `backbone_std` before passing it through the backbone. If False, the
            dummy image is left unchanged.

    Returns:
        The number of output channels produced by `backbone.forward_features`.
        This is the encoder feature dimension after global average pooling.
    """
    was_training = backbone.training
    backbone.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, backbone_input_hw[0], backbone_input_hw[1])

        if mobilenet_pretrained:
            mean = torch.tensor(backbone_mean, dtype=dummy.dtype).view(
                1, in_channels, 1, 1
            )
            std = torch.tensor(backbone_std, dtype=dummy.dtype).view(
                1, in_channels, 1, 1
            )
            dummy = (dummy - mean) / std

        feat = backbone.forward_features(dummy)
        encoder_dim = feat.shape[1]

    backbone.train(was_training)
    return encoder_dim


def build_cond_encoder(cond_dim: int) -> nn.Sequential:
    """
    Build the MLP that encodes the scalar conditioning value alpha.

    The generator receives alpha as a single value per image, usually with shape
    [B] or [B, 1]. Before alpha is combined with image features, it is projected
    into a higher-dimensional condition embedding of size `cond_dim`.

    The returned encoder has the layout:

        Linear(1, cond_dim) -> LeakyReLU
        -> Linear(cond_dim, cond_dim) -> LeakyReLU

    Args:
        cond_dim:
            Size of the condition embedding produced from the scalar alpha
            input.

    Returns:
        A small sequential MLP that maps alpha from shape [B, 1] to
        [B, cond_dim].
    """
    return nn.Sequential(
        nn.Linear(1, cond_dim),
        nn.LeakyReLU(),
        nn.Linear(cond_dim, cond_dim),
        nn.LeakyReLU(),
    )


# def flatten_output_params(params, batch_size):
#    default_order = TRANSFORMATIONS_IN_ORDER
# We did not use these settings!
# extra_order = {
#    "gamma": None,
#    "wb": None,
#    "bright": None,
#    "bw": None,
#    "hue": None,
#    "scale": None,
#    "affine": None,
# }

#    flat = []

# default params first
#    for key in default_order:
#        if key in params:
#            flat.append(params[key].reshape(batch_size, -1))

# extra params after
# for key in extra_order:
#    if key in params:
#        flat.append(params[key].reshape(batch_size, -1))

#    return torch.cat(flat, dim=1) if flat else torch.empty(batch_size, 0)


def normalize_for_mobilenet(x: torch.Tensor, dim: tuple[int, int]) -> torch.Tensor:
    """
    Encode the input image with the MobileNet backbone.

    The resize step is explicit so backbone preprocessing remains local and
    easy to inspect.
    """
    return F.interpolate(
        x,
        size=dim,
        mode="bicubic",  # see Mobilenet Definition
        align_corners=False,  # standard image-resizing; pixels treated like are rather than fixed corner points
        antialias=False,  # kept disabled for compatibility with export paths such as ONNX
    )


def piecewise_tanh_mapping(
    u: torch.Tensor,
    min_val: float,
    max_val: float,
    identity: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Map a latent value to a bounded parameter range with a piecewise tanh.

    This mapping preserves the identity point exactly:

        u = 0 -> identity

    Positive values move from identity toward max_val.
    Negative values move from identity toward min_val.

    This is useful when the valid parameter range is asymmetric around the
    identity value.

    Args:
        u:
            Latent edit tensor.

        min_val:
            Lower bound of the output parameter.

        max_val:
            Upper bound of the output parameter.

        identity:
            Neutral parameter value.

        scale:
            Controls how quickly the output saturates toward the bounds.
            Larger values make the mapping less sensitive.

    Returns:
        Bounded tensor with the same shape as u.
    """
    upper = max_val - identity
    lower = identity - min_val

    pos = identity + upper * torch.tanh(u / scale)
    neg = identity + lower * torch.tanh(u / scale)

    return torch.where(u >= 0, pos, neg)


def shifted_sigmoid_mapping(
    u: torch.Tensor,
    min_val: float,
    max_val: float,
    identity: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Map a latent value to a bounded parameter range with a shifted sigmoid.

    The sigmoid is shifted so that:

        u = 0 -> identity

    even when identity is not centered between min_val and max_val.

    Args:
        u:
            Latent edit tensor.

        min_val:
            Lower bound of the output parameter.

        max_val:
            Upper bound of the output parameter.

        identity:
            Neutral parameter value.

        eps:
            Small numerical constant used to avoid log or division issues when
            computing the sigmoid shift.

    Returns:
        Bounded tensor with the same shape as u.
    """
    min_tensor = torch.as_tensor(min_val, dtype=u.dtype, device=u.device)
    max_tensor = torch.as_tensor(max_val, dtype=u.dtype, device=u.device)
    identity_tensor = torch.as_tensor(identity, dtype=u.dtype, device=u.device)

    num = torch.clamp(identity_tensor - min_tensor, min=eps)
    den = torch.clamp(max_tensor - identity_tensor, min=eps)

    shift = torch.log(num / den)

    z = u + shift
    sigma = torch.sigmoid(z)

    return min_tensor + (max_tensor - min_tensor) * sigma


def validate_mapping_bounds(
    min_val: float,
    max_val: float,
    identity: float,
) -> None:
    """
    Validate that the identity value lies strictly inside the allowed range.

    Args:
        min_val:
            Lower bound of the output parameter.

        max_val:
            Upper bound of the output parameter.

        identity:
            Neutral parameter value.

    Raises:
        ValueError:
            If min_val < identity < max_val is not satisfied.
    """
    if not (min_val < identity < max_val):
        raise ValueError(
            f"Need min_val < identity < max_val, got "
            f"{min_val}, {identity}, {max_val}"
        )
