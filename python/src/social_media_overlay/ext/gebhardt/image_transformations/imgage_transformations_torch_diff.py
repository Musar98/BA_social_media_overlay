import torch
import numpy as np
from social_media_overlay.ext.gebhardt.image_transformations.color_transformations import rgb2hsv_torch, hsv2rgb_torch, rgb2lum, lerp

# https://github.com/christophgebhardt/regressor-guided-image-editing/blob/main/src/baselines/image_transformations/img_trans_torch_diff.py

def apply_curve_adjustment_patch(im, param, normalize=False):
    """
        Author: Marcel Schubert
    
        Fix onnx export error with overload:
        During symbolic export, i / curve_steps became a Python float (e.g. 0.125), but somehow PyTorchs FX picked an overload where min was treated as a fake tensor.
        The error says: "Expected a value of type Optional[number] for argument min but instead found type FakeTensor".
        This happens because the exporter tries different clamp overloads (scalar vs tensor arguments) and none match when FakeTensor appears
        
         rewrite the clamp calls so that all bounds are Python floats
    """
    curve_steps = param.shape[2]
    # Use new_zeros_like to get correct dtype/device
    total_image = im.new_zeros(im.size())
    param_list = torch.split(param, 1, dim=2)
    for i in range(curve_steps):
        # Ensure offset is a Python float
        offset = float(i) / float(curve_steps)
        # Perform clamp with explicit float bounds
        clamped = torch.clamp(im - offset, min=0.0, max=1.0/curve_steps)
        total_image = total_image + clamped * param_list[i]
    if normalize:
        color_curve_sum = param.sum(dim=2, keepdim=True) + 1e-9
        total_image = total_image * (curve_steps / color_curve_sum)
    else:
        # Also clamp final result with floats
        total_image = torch.clamp(total_image, min=0.0, max=1.0)
    return total_image

def apply_curve_adjustment(im, param, normalize=False):
    return apply_curve_adjustment_patch(im,param,normalize=normalize)
    """
    curve_steps = param.shape[2]
    total_image = im * 0
    param_list = torch.split(param, 1, dim=2)
    for i in range(curve_steps):
        total_image += torch.clamp(im - 1.0 * i / curve_steps, 0, 1.0 / curve_steps) * param_list[i]

    if normalize:
        color_curve_sum = param.sum(dim=2, keepdim=True) + 1e-9
        total_image *= curve_steps / color_curve_sum
    else:
        total_image = torch.clamp(total_image, max=1.0)

    return total_image
    """


def apply_saturation(im, saturation_param):
    # Convert to HSV
    hsv = rgb2hsv_torch(im)
    s = hsv[:, 1:2, :, :]
    v = hsv[:, 2:3, :, :]

    # Enhance saturation
    enhanced_s = s + (1 - s) * (0.5 - torch.abs(0.5 - v)) * 0.8

    # Combine back to HSV
    hsv1 = torch.cat([hsv[:, 0:1, :, :], enhanced_s, hsv[:, 2:, :, :]], dim=1)

    # Convert back to RGB
    full_color = torch.clamp(hsv2rgb_torch(hsv1), 0, 1)

    # Linearly interpolate between im and full_color using saturation_param
    saturated_im = lerp(im, full_color, saturation_param)

    return saturated_im


def apply_contrast(im, contrast_param):
    luminance = rgb2lum(im)
    contrast_lum = -torch.cos(np.pi * luminance) * 0.5 + 0.5
    contrast_image = im / (luminance + 1e-6) * contrast_lum
    contrast_image = torch.clamp(contrast_image, 0, 1)
    return lerp(im, contrast_image, contrast_param)


def apply_white_balance(im, white_balance_param):
    white_balance_param = white_balance_param.view(-1, 1, 1, 1)
    rgb_means = torch.mean(im, dim=(2, 3), keepdim=True) + 1e-9
    balancing_vec = 0.5 / rgb_means
    balancing_mat = balancing_vec.repeat(1, 1, im.size(2), im.size(3))
    wb_im = im * balancing_mat
    return torch.clamp(lerp(im, wb_im, white_balance_param), min=0, max=1)


def apply_exposure(im, exposure_param):
    exposure_param = exposure_param.view(-1, 1, 1, 1)
    exposed_im = im * torch.exp(exposure_param * torch.log(torch.tensor(2.0)))
    exposed_im = torch.clamp(exposed_im, min=0.0, max=1.0)
    return exposed_im


def apply_black_white(im, bw_param):
    bw_param = bw_param[:, None, None, None]
    luminance = rgb2lum(im)
    return lerp(im, luminance, bw_param)


def apply_gamma(im, gamma_param):
    gamma_param = gamma_param.repeat(1, im.shape[1], im.shape[2], im.shape[3])
    gamma_im = torch.pow(im + 1e-7, gamma_param)
    return gamma_im


def apply_sharpening(im, sharp_param, tf1, tf2):
    im1 = torch.nn.functional.conv2d(im, tf1, stride=1, padding='same')
    im2 = torch.nn.functional.conv2d(im, tf2, stride=1, padding='same')
    im_edges = torch.sqrt(torch.pow(im1, 2) + torch.pow(im2, 2) + 1e-7)
    # im_edges = torch.pow(im1, 2) + torch.pow(im2, 2)
    # im_edges = im1+im2
    sharp_param = sharp_param[:, :, None, None]
    sharp_param = sharp_param.repeat(1, im.shape[1], im.shape[2], im.shape[3])
    sharp_im = im + sharp_param * im_edges * im
    sharp_im = torch.clamp(sharp_im, min=0.0, max=1.0)
    return sharp_im