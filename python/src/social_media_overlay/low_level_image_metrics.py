from typing import Dict, List

import torch
import kornia.color as Kcolor
import torch.nn.functional as F

# Source: Adapted from Gebhardt. et. al.
# https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py
# IN THIS FILE ALL TENSORS ARE EXPECTED TO BE WITHIN [0.0, 1.0]!!!!!

def _check_images(images: torch.Tensor) -> torch.Tensor:
    """Validate, normalize dtype, and clamp a batch of RGB images.

    Args:
        images: Image batch as a PyTorch tensor with shape ``[B, 3, H, W]``.
            Pixel values are expected to be in the range ``[0.0, 1.0]``.
            Non-floating tensors are converted to ``float32``.

    Returns:
        A floating-point tensor with the same shape as ``images``, clamped to
        the valid image range ``[0.0, 1.0]``.

    Raises:
        TypeError: If ``images`` is not a ``torch.Tensor``.
        ValueError: If ``images`` does not have shape ``[B, 3, H, W]``.
    """

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images must be a torch.Tensor, got {type(images)}")

    # Metrics in this module expect batched RGB tensors in channel-first format.
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected [B, 3, H, W], got {tuple(images.shape)}")

    # Convert integer or boolean image tensors before clamping/computing metrics.
    # Note: this does not rescale 0-255 inputs to 0-1; it only changes dtype.
    if not torch.is_floating_point(images):
        images = images.float()

    return images.clamp(0.0, 1.0)


def calculate_colorfulness(images: torch.Tensor) -> torch.Tensor:
    """Calculate the colorfulness of a batch of RGB images.

    This is a batched PyTorch/Kornia version of the original PIL/NumPy
    implementation in ``low_level_image_metrics.py``. The metric follows
    Hasler and Suesstrunk's colorfulness measure, using the spread of pixel
    values in the LAB ``a*`` and ``b*`` channels.

    The original implementation computes:

        colorfulness = std(color_diff) + 0.3 * mean(color_diff)

    where ``color_diff`` is the Euclidean distance of each pixel's ``a*`` and
    ``b*`` values from their image-wise means.

    Difference from the reference implementation:
        The original function accepts a single image path, opens the image with
        PIL, converts it to LAB, and returns one scalar colorfulness value.
        This implementation accepts an already-loaded tensor batch with shape
        ``[B, 3, H, W]`` and returns one colorfulness score per image as a
        tensor of shape ``[B]``.

        The original implementation uses NumPy's ``np.std``, which computes the
        population standard deviation by default. To match that behavior, this
        implementation uses ``torch.std(..., unbiased=False)``.

        The original implementation calculates ``a_mean``, ``a_std``,
        ``b_mean``, and ``b_std``, but only the means are used in the final
        colorfulness metric. This implementation therefore only computes the
        values needed for the final formula.

    Args:
        images: Batch of RGB images as a tensor with shape ``[B, 3, H, W]``.
            Values are expected to be in ``[0.0, 1.0]``. Non-floating tensors
            are converted by ``_check_images``.

    Returns:
        A tensor of shape ``[B]`` containing one colorfulness score per image.

    Raises:
        TypeError: If ``images`` is not a ``torch.Tensor``.
        ValueError: If ``images`` does not have shape ``[B, 3, H, W]``.

    References:
        David Hasler and Sabine E. Suesstrunk, "Measuring colorfulness in
        natural images", Human Vision and Electronic Imaging VIII, 2003.

        Original reference implementation:
        https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py
    """
    images = _check_images(images)
    lab = Kcolor.rgb_to_lab(images)
    a = lab[:, 1]
    b = lab[:, 2]

    a_mean = a.mean(dim=(-2, -1), keepdim=True)
    b_mean = b.mean(dim=(-2, -1), keepdim=True)

    color_diff = torch.sqrt((a - a_mean) ** 2 + (b - b_mean) ** 2)
    mean_color_diff = color_diff.mean(dim=(-2, -1))
    std_color_diff = color_diff.std(dim=(-2, -1), unbiased=False)

    return std_color_diff + 0.3 * mean_color_diff


def compute_mean_brightness(images: torch.Tensor) -> torch.Tensor:
    """Compute the mean brightness of a batch of RGB images.

    This is a batched PyTorch/Kornia adaptation of the reference PIL/NumPy
    implementation in ``low_level_image_metrics.py``. The metric follows the
    mean brightness measure used in Goetschalckx et al., where brightness is
    computed as the average grayscale pixel value of an image.

    Difference from the reference implementation:
        The original function accepts a single image path, opens the image with
        PIL, converts it to grayscale using ``image.convert("L")``, and returns
        one scalar brightness value.

        This implementation accepts an already-loaded tensor batch with shape
        ``[B, 3, H, W]`` and returns one mean brightness value per image as a
        tensor with shape ``[B]``.

        The original PIL implementation typically works on 8-bit images and
        therefore returns brightness values in the range ``[0, 255]``. This
        implementation expects normalized image tensors in ``[0.0, 1.0]`` after
        ``_check_images`` and therefore returns brightness values in
        ``[0.0, 1.0]``.

        The grayscale conversion may also differ slightly from PIL because
        Kornia's ``rgb_to_grayscale`` uses tensor-based RGB-to-luminance
        conversion, while PIL's ``convert("L")`` uses PIL's internal grayscale
        conversion on image data.

    Args:
        images: Batch of RGB images as a tensor with shape ``[B, 3, H, W]``.
            Values are expected to be in ``[0.0, 1.0]``. Non-floating tensors
            are converted by ``_check_images``.

    Returns:
        A tensor with shape ``[B]`` containing one mean brightness score per
        image.

    Raises:
        TypeError: If ``images`` is not a ``torch.Tensor``.
        ValueError: If ``images`` does not have shape ``[B, 3, H, W]``.

    References:
        Goetschalckx et al., "GANalyze: Toward Visual Definitions of Cognitive
        Image Properties", ICCV 2019.

        Original reference implementation:
        https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py
    """
    images = _check_images(images)
    gray = Kcolor.rgb_to_grayscale(images)
    return gray.mean(dim=(1, 2, 3))


def compute_mean_saturation(images: torch.Tensor) -> torch.Tensor:
    """Compute the mean saturation of a batch of RGB images.

    This is a batched PyTorch/Kornia adaptation of the reference PIL/NumPy
    implementation in ``low_level_image_metrics.py``. The metric follows the
    saturation measure mentioned by Hasler and Suesstrunk, where image
    saturation is computed as the average value of the saturation channel in
    HSV color space.

    Difference from the reference implementation:
        The original function accepts a single image path, opens the image with
        PIL, converts it to HSV using ``image.convert("HSV")``, extracts the
        saturation channel, and returns one scalar mean saturation value.

        This implementation accepts an already-loaded tensor batch with shape
        ``[B, 3, H, W]`` and returns one mean saturation value per image as a
        tensor with shape ``[B]``.

        The original PIL implementation typically works on 8-bit images and
        therefore returns saturation values in the range ``[0, 255]``. This
        implementation expects normalized image tensors in ``[0.0, 1.0]`` after
        ``_check_images``. Kornia's ``rgb_to_hsv`` returns saturation in
        ``[0.0, 1.0]``, so this function returns mean saturation values in
        ``[0.0, 1.0]``.

        The HSV conversion may also differ slightly from PIL because Kornia
        performs a tensor-based RGB-to-HSV conversion, while PIL uses its own
        image conversion rules on image data.

    Args:
        images: Batch of RGB images as a tensor with shape ``[B, 3, H, W]``.
            Values are expected to be in ``[0.0, 1.0]``. Non-floating tensors
            are converted by ``_check_images``.

    Returns:
        A tensor with shape ``[B]`` containing one mean saturation score per
        image.

    Raises:
        TypeError: If ``images`` is not a ``torch.Tensor``.
        ValueError: If ``images`` does not have shape ``[B, 3, H, W]``.

    References:
        David Hasler and Sabine E. Suesstrunk, "Measuring Colourfulness in
        Natural Images", Human Vision and Electronic Imaging VIII, 2003.

        Original reference implementation:
        https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py
    """
    images = _check_images(images)
    hsv = Kcolor.rgb_to_hsv(images)
    s = hsv[:, 1]
    return s.mean(dim=(-2, -1))


def compute_rms_contrast(images: torch.Tensor) -> torch.Tensor:
    """Compute the RMS contrast of a batch of RGB images.

    This is a batched PyTorch/Kornia adaptation of the reference PIL/NumPy
    implementation in ``low_level_image_metrics.py``. The metric follows Peli's
    RMS contrast measure for complex images, where contrast is computed as the
    standard deviation of grayscale pixel intensities.

    Difference from the reference implementation:
        The original function accepts a single image path, opens the image with
        PIL, converts it to grayscale using ``image.convert("L")``, and returns
        one scalar RMS contrast value.

        This implementation accepts an already-loaded tensor batch with shape
        ``[B, 3, H, W]`` and returns one RMS contrast value per image as a
        tensor with shape ``[B]``.

        The original PIL/NumPy implementation typically works on 8-bit images
        and therefore returns contrast values on a ``[0, 255]`` intensity scale.
        This implementation expects normalized image tensors in ``[0.0, 1.0]``
        after ``_check_images`` and therefore returns contrast values on a
        ``[0.0, 1.0]`` intensity scale.

        The original implementation uses NumPy's ``np.std``, which computes the
        population standard deviation by default. To match that behavior, this
        implementation uses ``torch.std(..., unbiased=False)``.

        The grayscale conversion may also differ slightly from PIL because
        Kornia's ``rgb_to_grayscale`` uses tensor-based RGB-to-luminance
        conversion, while PIL's ``convert("L")`` uses PIL's internal grayscale
        conversion on image data.

    Args:
        images: Batch of RGB images as a tensor with shape ``[B, 3, H, W]``.
            Values are expected to be in ``[0.0, 1.0]``. Non-floating tensors
            are converted by ``_check_images``.

    Returns:
        A tensor with shape ``[B]`` containing one RMS contrast score per image.

    Raises:
        TypeError: If ``images`` is not a ``torch.Tensor``.
        ValueError: If ``images`` does not have shape ``[B, 3, H, W]``.

    References:
        Eli Peli, "Contrast in Complex Images", Journal of the Optical Society
        of America A, 7(10), 2032-2040, 1990.

        Original reference implementation:
        https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py
    """
    images = _check_images(images)
    gray = Kcolor.rgb_to_grayscale(images)
    return gray.flatten(1).std(dim=1, unbiased=False)


def compute_lighting_diversity(images: torch.Tensor) -> torch.Tensor:
    """Compute lighting diversity as variation in LAB luminance.

    This is a batched PyTorch/Kornia adaptation of the reference PIL/NumPy
    implementation in ``low_level_image_metrics.py``. The reference code defines
    lighting diversity as the standard deviation of the LAB ``L*`` channel.

    Difference from the reference implementation:
        The original function accepts a single image path, opens the image with
        PIL, converts it to RGB and then LAB, extracts the ``L*`` channel, and
        returns one scalar standard deviation value.

        This implementation accepts an already-loaded tensor batch with shape
        ``[B, 3, H, W]`` and returns one lighting-diversity value per image as a
        tensor with shape ``[B]``.

        The original implementation uses NumPy's ``np.std``, which computes the
        population standard deviation by default. To match that behavior, this
        implementation uses ``torch.std(..., unbiased=False)``.

        The LAB conversion may differ slightly from the reference implementation
        because Kornia performs a tensor-based RGB-to-LAB conversion, while the
        original code relies on its own PIL/NumPy LAB conversion helpers.

    Args:
        images: Batch of RGB images as a tensor with shape ``[B, 3, H, W]``.
            Values are expected to be in ``[0.0, 1.0]``. Non-floating tensors
            are converted by ``_check_images``.

    Returns:
        A tensor with shape ``[B]`` containing one lighting-diversity score per
        image.

    Raises:
        TypeError: If ``images`` is not a ``torch.Tensor``.
        ValueError: If ``images`` does not have shape ``[B, 3, H, W]``.

    References:
        Original reference implementation:
        https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py
    """
    images = _check_images(images)
    lab = Kcolor.rgb_to_lab(images)
    l = lab[:, 0]
    return l.flatten(1).std(dim=1, unbiased=False)


def compute_blur_effect(images: torch.Tensor) -> torch.Tensor:
    """Compute a fast sharpness proxy for a batch of RGB images.

    This function uses the variance of the image Laplacian as a focus/sharpness
    measure. Blur removes high-frequency edge information, which usually leads
    to a lower Laplacian variance.

    Therefore:
        Higher values indicate sharper / less blurred images.
        Lower values indicate blurrier images.

    Note on difference from the reference implementation:
        The original function in ``low_level_image_metrics.py`` computes the
        no-reference perceptual blur metric described by Crete et al. using
        ``skimage.measure.blur_effect``. That metric returns a blur score where
        higher values indicate stronger blur.

        This implementation does not reproduce ``skimage.measure.blur_effect``.
        Instead, it computes a GPU-compatible PyTorch proxy based on the
        variance of the Laplacian. This is a sharpness measure, not a direct
        perceptual blur-effect measure.

        As a result, the direction of the score is reversed compared with the
        reference blur-effect metric:

            Reference Crete / skimage blur_effect:
                higher value -> blurrier image
                lower value  -> sharper image

            This Laplacian-variance proxy:
                higher value -> sharper / less blurred image
                lower value  -> blurrier image

        The absolute values are therefore not comparable to the reference
        implementation. The metric should only be used as an efficient proxy
        when relative sharpness comparisons are sufficient.

    Difference in input and output format:
        The original function accepts a single image path, opens the image with
        PIL, converts it to grayscale, and returns one scalar blur-effect score.

        This implementation accepts an already-loaded tensor batch with shape
        ``[B, 3, H, W]`` and returns one sharpness score per image as a tensor
        with shape ``[B]``.

        The original PIL/skimage implementation usually operates on 8-bit
        grayscale images, while this implementation expects normalized image
        tensors in ``[0.0, 1.0]`` after ``_check_images``.

    Args:
        images: Batch of RGB images as a tensor with shape ``[B, 3, H, W]``.
            Values are expected to be in ``[0.0, 1.0]``. Non-floating tensors
            are converted by ``_check_images``.

    Returns:
        A tensor with shape ``[B]`` containing one Laplacian-variance sharpness
        score per image.

    Raises:
        TypeError: If ``images`` is not a ``torch.Tensor``.
        ValueError: If ``images`` does not have shape ``[B, 3, H, W]``.

    References:
        Crete et al., "The Blur Effect: Perception and Estimation with a New
        No-Reference Perceptual Blur Metric", Proc. SPIE 6492, Human Vision and
        Electronic Imaging XII.

        Pertuz, Puig, and Garcia, "Analysis of focus measure operators for
        shape-from-focus", Pattern Recognition, 46(5), 1415-1432, 2013.

        Original reference implementation:
        https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/analysis/low_level_image_metrics.py

        skimage blur_effect example:
        https://scikit-image.org/docs/stable/auto_examples/filters/plot_blur_effect.html
    """
    images = _check_images(images)
    gray = Kcolor.rgb_to_grayscale(images)  # [B, 1, H, W]

    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3)

    lap = F.conv2d(gray, lap_kernel, padding=1)  # [B, 1, H, W]
    sharpness = lap.flatten(1).var(dim=1, unbiased=False)

    return sharpness


def val_image_properties(images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    images: [B, 3, H, W] in [0,1]

    Returns per-image metric values, each of shape [B].
    You can append this dict yourself batch by batch.
    """
    images = _check_images(images)

    return {
        "colorfulness": calculate_colorfulness(images),
        "mean_brightness": compute_mean_brightness(images),
        "mean_saturation": compute_mean_saturation(images),
        "rms_contrast": compute_rms_contrast(images),
        "lighting_diversity": compute_lighting_diversity(images),
        "blur_effect": compute_blur_effect(images),
    }


def summarize_image_property_dicts(
    metric_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """
    metric_dicts: list of outputs from val_image_properties(...)

    Returns aggregated statistics:
    {
        "colorfulness_mean": ...,
        "colorfulness_std": ...,
        ...
        "num_samples": ...
    }
    """
    if len(metric_dicts) == 0:
        return {
            "colorfulness_mean": 0.0,
            "colorfulness_std": 0.0,
            "mean_brightness_mean": 0.0,
            "mean_brightness_std": 0.0,
            "mean_saturation_mean": 0.0,
            "mean_saturation_std": 0.0,
            "rms_contrast_mean": 0.0,
            "rms_contrast_std": 0.0,
            "lighting_diversity_mean": 0.0,
            "lighting_diversity_std": 0.0,
            "blur_effect_mean": 0.0,
            "blur_effect_std": 0.0,
            "num_samples": 0,
        }

    metric_names = metric_dicts[0].keys()
    result = {}

    num_samples = None

    for name in metric_names:
        values = [d[name].detach().reshape(-1).float().cpu() for d in metric_dicts]
        values = torch.cat(values, dim=0)

        result[f"{name}_mean"] = values.mean().item()
        result[f"{name}_std"] = values.std(unbiased=False).item()

        if num_samples is None:
            num_samples = values.numel()

    result["num_samples"] = 0 if num_samples is None else num_samples
    return result