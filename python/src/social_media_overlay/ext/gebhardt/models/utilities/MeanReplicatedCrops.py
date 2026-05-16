import torch.nn as nn

# https://arxiv.org/pdf/2501.12289
class MeanReplicatedCrops(nn.Module):
    """
    Averages the replicated crops for each original image in the transformed tensor,
    reducing the batch size back to its original.
    """

    def __init__(self, num_replications=10):
        """
        Constructor
        :param num_replications: The number of times each image should be replicated and cropped.
        """
        super(MeanReplicatedCrops, self).__init__()
        self.num_replications = num_replications

    def forward(self, x):
        """
        Forward pass of the module.
        :param x: The torch.Tensor with shape (B*num_replications, C, H, W) from ReplicateAndCrop module.
        returns: he meaned torch.Tensor with the original shape (B, C, H, W).
        """
        b, w = x.size()
        batch_size = b // self.num_replications
        result = x.view(batch_size, self.num_replications, w).mean(dim=1)
        return result
