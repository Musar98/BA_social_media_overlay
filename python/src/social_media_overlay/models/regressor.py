from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from social_media_overlay.ext.gebhardt.models.utilities.MeanReplicatedCrops import (
    MeanReplicatedCrops,
)
from social_media_overlay.ext.gebhardt.models.utilities.ReplicateAndCrop import (
    ReplicateAndCrop,
)

# Adapted from Gebhardt. et. al.

# Number of regression outputs produced by the final ResNet layer.
# The model predicts both mean and uncertainty values:
#   0 = valence_mean
#   1 = arousal_mean
#   2 = valence_std
#   3 = arousal_std
NUM_OUTPUT_CLASSES = 4  # [valence_mean, arousal_mean, valence_std, arousal_std]

# Number of crops created during ten-crop inference.
# This must match the replication count expected by ReplicateAndCrop and
# MeanReplicatedCrops, otherwise crop aggregation will be incorrect.
NUM_TEN_CROP_REPLICATIONS = 10

# Channel-wise mean used for normalizing RGB images.
# The values assume that image tensors are scaled to [0, 1] before normalization.
# Using mean 0.5 maps the center of the [0, 1] range to 0.
NORMALIZE_MEAN = (0.5, 0.5, 0.5)

# Channel-wise standard deviation used for normalizing RGB images.
# Together with NORMALIZE_MEAN, this maps inputs from [0, 1] approximately
# to [-1, 1], which is the expected input range for this model setup.
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Default resize dimension applied before cropping.
# Input images are first resized to this spatial size, then cropped to
# DEFAULT_CROP_SIZE. This follows the preprocessing used by the original model.
DEFAULT_INPUT_SIZE = 480

# Default crop size used after resizing.
# The model was trained/evaluated with 448x448 crops, so changing this value may
# affect prediction quality unless the model was trained with another crop size.
DEFAULT_CROP_SIZE = 448


# Adapted from Gebhardt. et. al.
class EmotionRegressor:
    """
    Predict valence and arousal scores from input images using a ResNet50 backbone.

    The underlying model predicts four values per image:

        [valence_mean, arousal_mean, valence_std, arousal_std]

    Depending on the selected `loss` mode, only a subset of these outputs is
    returned by `predict`.

    Args:
        path_to_model: Path to the pretrained ResNet50 state dictionary.
        average: If True, use ten-crop inference and average predictions across crops.
        normalize: If True, normalize input images from [0, 1] to approximately [-1, 1].
            If False, inputs are assumed to already be in the expected range.
        loss: Output selection mode. Supported values are:
            "va": return valence and arousal.
            "valence": return valence only.
            "arousal": return arousal only.
        input_size: Resize dimension applied before cropping. If None, resizing is skipped.
        crop_size: Crop size applied after resizing. If None, cropping is skipped.
        requires_grad: If False, freeze all parameters of the ResNet50 model.
        device: Device on which the model should be loaded and executed.

    Attributes:
        is_minimized: Compatibility flag used by the surrounding model code.
        device: Device assigned to the model.
        output_ixs: Output indices selected according to the chosen `loss` mode.
        model: Complete preprocessing and inference pipeline.
    """

    def __init__(
        self,
        path_to_model: str,
        average: bool,
        normalize: bool,
        loss: str = "va",
        input_size: Optional[int] = DEFAULT_INPUT_SIZE,
        crop_size: Optional[int] = DEFAULT_CROP_SIZE,
        requires_grad: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:

        self.is_minimized = True
        self.device = device

        activation_function = torch.nn.Sigmoid()
        is_ten_crop = average

        # Initialize the ResNet50 backbone.
        # The default ImageNet classification head is replaced with a linear
        # regression head that outputs four emotion-related values.
        modelresnet50 = models.resnet50()
        num_ft = modelresnet50.fc.in_features
        modelresnet50.fc = nn.Linear(num_ft, NUM_OUTPUT_CLASSES)

        # Load pretrained weights.
        # map_location ensures that weights can be loaded even if they were saved
        # on a different device than the one currently used.
        modelresnet50.load_state_dict(torch.load(path_to_model, map_location=device))

        # Optionally freeze the backbone and regression head.
        # This is useful when the regressor is used only for inference or as a
        # fixed feature/prediction module inside a larger system.
        if not requires_grad:
            if hasattr(modelresnet50, "parameters"):
                for p in modelresnet50.parameters():
                    p.requires_grad = False

        modelresnet50.eval()

        # Build preprocessing and inference pipeline.
        # The order is important:
        #   1. resize image
        #   2. crop image
        #   3. normalize image if needed
        #   4. run ResNet50
        #   5. aggregate ten-crop predictions if needed
        #   6. apply sigmoid activation
        modules = []
        # Resize input images to the configured size before cropping.
        # This step is skipped when input_size is None.
        if input_size is not None:
            modules.append(transforms.Resize(input_size, antialias=True))
        # Apply either center crop or ten-crop replication.
        # In single-crop mode, one centered crop is used.
        # In ten-crop mode, ReplicateAndCrop creates several crops per image,
        # which are later averaged by MeanReplicatedCrops.
        if crop_size is not None:
            if not is_ten_crop:
                modules.append(transforms.CenterCrop(crop_size))
            else:
                modules.append(
                    ReplicateAndCrop(crop_size, normalize, NUM_TEN_CROP_REPLICATIONS)
                )

        # Normalize only in single-crop mode.
        # In ten-crop mode, normalization is handled inside ReplicateAndCrop so
        # that each replicated crop is normalized consistently.
        if normalize and not is_ten_crop:
            modules.append(transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD))
        # Add the actual neural network after preprocessing.
        modules.append(modelresnet50)

        # Average predictions over the replicated crops.
        # This is only used when ten-crop evaluation is enabled.
        if is_ten_crop:
            modules.append(MeanReplicatedCrops(NUM_TEN_CROP_REPLICATIONS))

        # Apply sigmoid to constrain outputs to the expected [0, 1] range.
        if activation_function is not None:
            modules.append(activation_function)

        model = nn.Sequential(*modules)

        # Select which output dimensions should be returned by predict().
        # Full model output:
        #   0 = valence_mean
        #   1 = arousal_mean
        #   2 = valence_std
        #   3 = arousal_std
        #
        # The standard "va" mode returns only the two mean predictions.
        if loss == "valence":
            self.output_ixs = [0]
        elif loss == "arousal":
            self.output_ixs = [1]
        else:
            self.output_ixs = [0, 1]

        self.model = model.to(device)

    def to(self, device: torch.device) -> "EmotionRegressor":
        """
        Move the model pipeline to a specific device.

        Args:
            device: Target device, for example `torch.device("cuda")` or
                `torch.device("cpu")`.

        Returns:
            The current EmotionRegressor instance, allowing chained calls.
        """
        self.model.to(device)
        return self

    def eval(self) -> "EmotionRegressor":
        """
        Set the model pipeline to evaluation mode.

        Returns:
            The current EmotionRegressor instance, allowing chained calls.
        """
        self.model.eval()
        return self

    def train(self) -> "EmotionRegressor":
        """
        Set the model pipeline to training mode.

        Returns:
            The current EmotionRegressor instance, allowing chained calls.
        """
        self.model.train()
        return self

    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the model and return the selected outputs.

        Args:
            imgs: Input image tensor with shape `(B, C, H, W)`, where:
                B = batch size
                C = number of channels
                H = image height
                W = image width

        Returns:
            Tensor containing the selected predictions according to `loss`.
            Shape is `(B, 2)` for "va" mode and `(B, 1)` for "valence" or
            "arousal" mode.
        """
        return self.model(imgs)[:, self.output_ixs]
