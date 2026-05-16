import torch
import math
import torch.nn as nn
from image_transformations.image_transformations import apply_params


class ParametricGenerator(nn.Module):
    """
    Generator that learns a parametric global transformation of images
    """

    def __init__(self, num_features, fc_dim, num_conv_layers=7, params=None, do_norm=True, in_channels=3):
        """
        Constructor
        :param num_features: number of output features of first conv layer
        :param fc_dim: number of output features of fully connected layers
        :param num_conv_layers: number of convolutional layers (default=7)
        :param params: image transformation parameters the generator is supposed to predict and apply
        (default is specified in constructor)
        :param do_norm: flag indicating if batch norm is applied or not (default=True)
        :param in_channels: number of input channels of first conv layer (default=3)
        """
        super(ParametricGenerator, self).__init__()
        self.num_features = num_features
        self.do_norm = do_norm

        self.fc_dim = fc_dim
        self.curve_steps = 8
        self.color_channels = 3

        # last list of all generated images
        self.img_list = None
        
        # parameters the generator is supposed to predict,
        # order in list also defines order of explicit image transformations
        if params is None:
            # params look here:
            # params = ['sharp', 'exposure', 'contrast', 'tone', 'color']
            # transformations inspired by theory (tone and color to approximate lighting conditions) ordered low to high
            params = ['exposure', 'saturation', 'tone', 'color', 'contrast', 'sharp', 'blur']
            # all params, ordered low to high
            # params = ['exposure', 'bright', 'gamma', 'bw', 'wb', 'hue', 'color',
            #           'saturation', 'contrast', 'sharp', 'blur', 'tone', 'scale']

        # define convolutional layers
        self.conv_layers = nn.ModuleList()
        out_channels_factor = 1
        for i in range(num_conv_layers):
            if i == 0:
                padding = 3
                kernel_size = 7
            else:
                padding = 1
                kernel_size = 3
            self.conv_layers += [
                self.general_conv2d(in_channels=in_channels, out_channels=self.num_features * out_channels_factor,
                                    kernel_size=kernel_size, stride=2, padding=padding)]
            in_channels = self.num_features * out_channels_factor
            out_channels_factor *= 2

        # define fully connected layers
        out_channels_factor = int(out_channels_factor / 2)  # revert to get output channels of conv layers
        self.dense0 = nn.Linear(in_features=self.num_features * out_channels_factor, out_features=fc_dim)
        self.dense1 = nn.Linear(in_features=fc_dim, out_features=fc_dim)
        self.param_heads = self.define_parameter_prediction_heads(fc_dim, params)

    def define_parameter_prediction_heads(self, fc_dim, params):
        """
        Defines the parameter prediction heads according to the passed parameters and returns them in a dict. It uses 
        activation functions to attain parameters in a meaningful range given the explicit image transformation.
        :param fc_dim: number of input features of prediction heads
        :param params: list of parameters
        :return: dict of prediction heads for parameters
        """
        param_heads = nn.ModuleDict()
        for param in params:
            if param == 'affine':
                head_trans = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=2), nn.Tanh(),
                                           nn.Unflatten(1, (2, 1)), Multiply(100.0))
                head_rot = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=4), nn.Tanh(),
                                         nn.Unflatten(1, (2, 2)), Multiply(0.25))
                param_heads["trans"] = head_trans
                param_heads["rot"] = head_rot
                continue

            head = None
            if param == 'gamma':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Sigmoid(), Multiply(2.0))
            elif param == 'sharp':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Sigmoid(),
                                     nn.Flatten(start_dim=0), Multiply(15.0))
            elif param == 'wb':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Tanh(), Multiply(10.0))
            elif param == 'exposure':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Tanh(),
                                     nn.Flatten(start_dim=0), Multiply(3.0))
            elif param == 'bright':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Sigmoid())
            elif param == 'contrast':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Sigmoid(),
                                     nn.Flatten(start_dim=0), Multiply(3.0))
            elif param == 'saturation':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Sigmoid(), Multiply(10.0))
            elif param == 'bw':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1),
                                     nn.Sigmoid(), nn.Flatten(start_dim=0))
            elif param == 'tone':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=self.curve_steps), nn.Sigmoid(),
                                     nn.Unflatten(1, (1, self.curve_steps, 1)), Multiply(3.0))
            elif param == 'color':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=self.curve_steps * 3), nn.Sigmoid(),
                                     nn.Unflatten(1, (3, self.curve_steps, 1)), Multiply(3.0))
            elif param == 'blur':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Sigmoid(),
                                     nn.Flatten(start_dim=0), Multiply(10.0))
            elif param == 'hue':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=1), nn.Tanh(), Multiply(math.pi))

            elif param == 'scale':
                head = nn.Sequential(nn.Linear(in_features=fc_dim, out_features=2), nn.Sigmoid(),
                                     nn.Unflatten(1, (2, )), Multiply(3.0))

            if head is not None:
                param_heads[param] = head
                
        return param_heads

    def forward(self, x):
        """
        Forward pass
        :param x: image
        :return: list of generated images
        """
        enc = self.get_img_enc(x)
        params = self.get_predicted_params(enc)
        self.img_list = apply_params(x[:, :self.color_channels, :, :], params)
        return self.img_list[-1], params

    def get_img_enc(self, x):
        """
        Returns encoding of image after convolutional layers
        :param x: image
        :return:
        """
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
        x = torch.mean(x, dim=[2, 3])
        return x

    def get_predicted_params(self, x):
        """
        Returns prediction of parameters
        :param x: image encoding
        :return: parameters
        """
        x = nn.functional.leaky_relu(self.dense0(x))
        x = nn.functional.leaky_relu(self.dense1(x))
        
        dict_params = {}
        for param, head in self.param_heads.items():
            # constructing affine transformation param
            if param == "trans" or param == "rot":
                if 'affine' in dict_params:
                    continue
                # translation
                translation_param = self.param_heads["trans"](x)
                # rotation
                matrix_eye = torch.eye(2, 2)[None].repeat(x.shape[0], 1, 1).to(x.device)
                rotation_param = self.param_heads["rot"](x) + matrix_eye
                dict_params['affine'] = torch.concat((rotation_param, translation_param), dim=2)
            else:
                dict_params[param] = head(x)
            
        return dict_params

    @staticmethod
    def general_conv2d(in_channels, out_channels=64, kernel_size=3, stride=1, padding="valid", do_norm=True):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(num_features=out_channels) if do_norm else nn.Identity()
        )


class Multiply(nn.Module):
    """
    Simple module to multiply the output in a nn.Sequential.
    """
    def __init__(self, alpha):
        """
        Constructor
        :param alpha: factor of multiplication
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        x = torch.mul(x, self.alpha)
        return x