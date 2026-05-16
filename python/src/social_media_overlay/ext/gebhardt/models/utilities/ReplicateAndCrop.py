import torch
import torch.nn as nn
from torchvision import transforms

# https://arxiv.org/pdf/2501.12289
class ReplicateAndCrop(nn.Module):
    """
    Replicates each image in a batch tensor num_replications times and applies
    a random center crop to each replication. The replications are added to the batch size.
    """
    def __init__(self, crop_size=448, normalize=False, num_replications=10):
        """
        Constructor
        :param crop_size: The size (crop_size x crop_size) of the cropped image (in pixels).
        :param normalize: Normalize to -1 and 1 (default = False)
        :param num_replications: The number of times each image should be replicated and cropped.
        """
        super(ReplicateAndCrop, self).__init__()
        self.crop_size = crop_size
        self.num_replications = num_replications
        transform_list = [
            # transforms.ToPILImage(),
            transforms.RandomCrop(crop_size),
            # transforms.ToTensor()
        ]
        if normalize:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform_list)

    def forward(self, x):
        """
        Forward pass of the module.
        :param x: A tensor of shape (B, C, H, W) representing a batch of images.
        returns: The transformed tensor with an increased batch size (B*num_replications, C, H, W).
        """
        batch_size, c, h, w = x.size()
        output = torch.empty((batch_size * self.num_replications, c, self.crop_size, self.crop_size),
                             dtype=x.dtype, device=x.device)

        for i in range(batch_size):
            img = x[i]
            for j in range(self.num_replications):
                output[i * self.num_replications + j] = self.transform(img)

        return output
