import torch
import kornia

# run patch
import social_media_overlay.ext.gebhardt.image_transformations.kornia_monkey_patch

import social_media_overlay.ext.gebhardt.image_transformations.imgage_transformations_torch_diff as ittf
import math

# https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/baselines/image_transformations/image_transformations.py

def apply_params(im, params):
    """
    Applies parameters to image.
    :param im: image
    :param params: predicted parameters
    :return: list of transformed images
    """
    param_names = list(params.keys())
    im_list = []
    for i in range(len(param_names)):
        # Apply Gamma
        if 'gamma' == param_names[i]:
            im = apply_gamma(im, params['gamma'])
        # Apply sharpening
        elif 'sharp' == param_names[i]:
            im = apply_sharpening(im, params['sharp'])
        # Apply WB
        elif "wb" == param_names[i]:
            im = apply_white_balance(im, params['wb'])
        # Apply exposure
        elif "exposure" == param_names[i]:
            im = ittf.apply_exposure(im, params['exposure'])
        # Apply brightness
        elif "bright" == param_names[i]:
            im = apply_brightness(im, params['bright'])
        # Apply contrast
        elif "contrast" == param_names[i]:
            im = apply_contrast(im, params['contrast'])
        # Apply saturation
        elif "saturation" == param_names[i]:
            im = apply_saturation(im, params['saturation'])
        # Apply black and white
        elif "bw" == param_names[i]:
            im = apply_black_white(im, params['bw'])
        # Apply tone curve adjustment
        elif "tone" == param_names[i]:
            im = apply_tone_curve_adjustment(im, params['tone'])
        # Apply hue
        elif "hue" == param_names[i]:
            im = apply_hue(im, params['hue'])
        # Apply color curve adjustment
        elif "color" == param_names[i]:
            im = apply_color_curve_adjustment(im, params['color'])
        # Apply blur
        elif "blur" == param_names[i]:
            im = apply_gaussian_blur(im, params['blur'])
        # Apply affine transformation
        elif "affine" == param_names[i]:
            im = apply_affine_transformation(im, params['affine'])
        # Apply scale transformation
        elif "scale" == param_names[i]:
            im = apply_scale(im, params['scale'])

        im = torch.clamp(im, min=0.0, max=1.0)
        if i == len(param_names) - 1:
            im_list.append(im)
        else:
            im_list.append(im.detach().clone())

    return im_list


def apply_color_curve_adjustment(im, color_param):
    """
    Adjusts color of image according to passed parameter tensor
    :param im: image tensors
    :param color_param: parameter tensor of shape (:, 3, curve_steps, 1) where curve_steps the number of curve steps,
    defined from 0 to 3
    :return: color adjusted image
    """
    return ittf.apply_curve_adjustment(im, color_param)


def apply_tone_curve_adjustment(im, tone_param):
    """
    Adjusts color tone of image according to passed parameter tensor
    :param im: image tensors
    :param tone_param: parameter tensor of shape (:, 1, curve_steps, 1) where curve_steps the number of curve steps,
    defined from 0 to 3
    :return: tone adjusted image
    """
    return ittf.apply_curve_adjustment(im, tone_param)

# probably this is fucked?
def apply_saturation(im, saturation_param):
    """
    Increases decreases saturation of image.
    :param im: image tensors
    :param saturation_param: factor of saturation increase, defined from 0 to inf
    :return: saturation adjusted image tensors
    """
    return kornia.enhance.adjust_saturation(im, torch.clamp(saturation_param, min=0))


def apply_contrast(im, contrast_param):
    """
    Increases decreases contrast of image.
    :param im: image tensors
    :param contrast_param: factor of contrast increase, defined from 0 to 1
    :return: contrast adjusted image tensors
    """
    # return kornia.enhance.adjust_contrast(im, torch.clamp(contrast_param, min=0), clip_output=True)
    return kornia.enhance.adjust_contrast_with_mean_subtraction(im, contrast_param)


def apply_gaussian_blur(im, blur_param, kernel_size=(25, 25)):
    """
    Blurring the image tensors.
    :param im: image tensors
    :param blur_param: standard deviation of blur kernel, defined from 0 to inf
    :param kernel_size: kernel size, default=(25, 25), which is the same as in look here
    :return: Blurred images
    """
    sigma = torch.clamp(blur_param, min=0)
    sigma = sigma.repeat(2, 1).view(-1, 2)
    im = kornia.filters.gaussian_blur2d(im, kernel_size, sigma)
    return torch.clamp(im, min=0.0, max=1.0)


def apply_white_balance(im, white_balance_param):
    """
    Changing white balance of image.
    :param im: image tensors
    :param white_balance_param: parameter tensor
    :return: Adjusted images
    """
    return ittf.apply_white_balance(im, white_balance_param)


def apply_brightness(im, brightness_param):
    """
    Brightness adjustment of image.
    :param im: image tensors
    :param brightness_param: tensor of brightness factor, defined from 0 to 1.
    :return: Brightened image
    """
    return kornia.enhance.adjust_brightness(im, torch.clamp(brightness_param, min=0, max=1), clip_output=True)


def apply_exposure(im, exposure_param):
    """
    Exposure adjustment of image.
    :param im: image tensors
    :param exposure_param: tensor of exposure factor, defined from 0 to 1.
    :return: Brightened image
    """
    return ittf.apply_exposure(im, exposure_param)


def apply_black_white(im, bw_param):
    """
    Black and white filter of image.
    :param im: image tensors
    :param bw_param: parameter tensor deciding between black and white and all colored
    :return: adjusted images
    """
    return ittf.apply_black_white(im, bw_param)


def apply_hue(im, hue_param):
    """
    Hue adjustment of image tensors.
    :param im: image tensors
    :param hue_param: tensor of hue factor, defined from -pi to pi
    :return: Hue adjusted images
    """
    return kornia.enhance.adjust_hue(im, torch.clamp(hue_param, min=-math.pi, max=math.pi))


def apply_gamma(im, gamma_param):
    """
    Image gamma transformation.
    :param im: image tensors
    :param gamma_param: tensor of gama factor. Non-negative real number. gamma larger than 1 make the shadows darker,
    while gamma smaller than 1 make dark
    regions lighter.
    :return: enhanced image
    """
    return kornia.enhance.adjust_gamma(im, torch.clamp(gamma_param, min=0), gain=1.0)


def apply_sharpening(im, sharp_param):
    """
    Image sharpening
    :param im: image tensors
    :param sharp_param: tensor of sharp factor, defined from 0 to inf
    :return: Sharpened image
    """
    return kornia.enhance.sharpness(im, torch.clamp(sharp_param, min=0))


def apply_affine_transformation(im, matrices):
    """
    Affine transformation
    :param im: image tensors
    :param matrices: tensor of 2x3 2D transformation matrices
    :return: Transformed images
    """
    im = kornia.geometry.transform.affine(im, matrices, padding_mode='border')
    return torch.clamp(im, min=0.0, max=1.0)


def apply_scale(im, scale_param):
    """
    Affine transformation
    :param im: image tensors
    :param scale_param: tensor of bx1x2 or bx1x4 depending on whether center specified or not.
    :return: Transformed images
    """
    center = None
    if scale_param.size(1) == 4:
        center = scale_param[:, 2:4]
    scale_factor = scale_param[:, 0:2]

    return kornia.geometry.transform.scale(im, scale_factor, center=center)