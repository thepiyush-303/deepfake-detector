import numpy as np
import torch
import torch.nn as nn


def get_srm_kernels():
    """
    Define all 30 fixed SRM filter kernels (5x5).
    Categories:
    - 1st-order edge (6)
    - 2nd-order (4)
    - 3rd-order (8)
    - Square/Laplacian (4)
    - Edge+diagonal (8)
    
    All filters sum to 0 for content suppression.
    
    Returns:
        numpy array of shape (30, 5, 5)
    """
    kernels = [
        # 1st-order edge filters (6 filters)
        [[ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0], [-1, -1,  4, -1, -1], [ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0]],
        [[ 0,  0, -1,  0,  0], [ 0,  0, -1,  0,  0], [ 0,  0,  4,  0,  0], [ 0,  0, -1,  0,  0], [ 0,  0, -1,  0,  0]],
        [[-1,  0,  0,  0,  0], [ 0, -1,  0,  0,  0], [ 0,  0,  4,  0,  0], [ 0,  0,  0, -1,  0], [ 0,  0,  0,  0, -1]],
        [[ 0,  0,  0,  0, -1], [ 0,  0,  0, -1,  0], [ 0,  0,  4,  0,  0], [ 0, -1,  0,  0,  0], [-1,  0,  0,  0,  0]],
        [[ 0,  0,  0,  0,  0], [-1, -1,  4, -1, -1], [ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0]],
        [[ 0, -1,  0,  0,  0], [ 0, -1,  0,  0,  0], [ 0,  4,  0,  0,  0], [ 0, -1,  0,  0,  0], [ 0, -1,  0,  0,  0]],
        
        # 2nd-order filters (4 filters)
        [[ 0,  0, -1,  0,  0], [ 0,  0,  2,  0,  0], [-1,  2, -4,  2, -1], [ 0,  0,  2,  0,  0], [ 0,  0, -1,  0,  0]],
        [[ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0], [ 1, -2,  2, -2,  1], [ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0]],
        [[ 0,  0,  1,  0,  0], [ 0,  0, -2,  0,  0], [ 0,  0,  2,  0,  0], [ 0,  0, -2,  0,  0], [ 0,  0,  1,  0,  0]],
        [[ 1,  0,  0,  0,  0], [ 0, -2,  0,  0,  0], [ 0,  0,  2,  0,  0], [ 0,  0,  0, -2,  0], [ 0,  0,  0,  0,  1]],
        
        # 3rd-order filters (8 filters)
        [[ 0,  0,  0,  0,  0], [ 0,  1, -2,  1,  0], [ 0, -2,  4, -2,  0], [ 0,  1, -2,  1,  0], [ 0,  0,  0,  0,  0]],
        [[ 0,  0,  1,  0,  0], [ 0, -2,  0, -2,  0], [ 1,  0,  4,  0,  1], [ 0, -2,  0, -2,  0], [ 0,  0,  1,  0,  0]],
        [[ 0,  1,  0,  1,  0], [ 1, -4,  0, -4,  1], [ 0,  0,  8,  0,  0], [ 1, -4,  0, -4,  1], [ 0,  1,  0,  1,  0]],
        [[ 1,  0, -2,  0,  1], [ 0, -2,  4, -2,  0], [-2,  4, -4,  4, -2], [ 0, -2,  4, -2,  0], [ 1,  0, -2,  0,  1]],
        [[-1,  2, -2,  2, -1], [ 2, -6,  8, -6,  2], [-2,  8,-12,  8, -2], [ 2, -6,  8, -6,  2], [-1,  2, -2,  2, -1]],
        [[ 0,  0, -1,  0,  0], [ 0,  2,  2,  2,  0], [-1,  2,-12,  2, -1], [ 0,  2,  2,  2,  0], [ 0,  0, -1,  0,  0]],
        [[ 0, -1,  2, -1,  0], [-1,  4, -6,  4, -1], [ 2, -6,  8, -6,  2], [-1,  4, -6,  4, -1], [ 0, -1,  2, -1,  0]],
        [[ 1, -2,  0, -2,  1], [-2,  4,  2,  4, -2], [ 0,  2,-12,  2,  0], [-2,  4,  2,  4, -2], [ 1, -2,  0, -2,  1]],
        
        # Square/Laplacian filters (4 filters)
        [[ 0,  0, -1,  0,  0], [ 0, -1,  4, -1,  0], [-1,  4, -8,  4, -1], [ 0, -1,  4, -1,  0], [ 0,  0, -1,  0,  0]],
        [[-1, -1, -1, -1, -1], [-1,  8,  8,  8, -1], [-1,  8,-48,  8, -1], [-1,  8,  8,  8, -1], [-1, -1, -1, -1, -1]],
        [[ 0, -1, -1, -1,  0], [-1,  4,  4,  4, -1], [-1,  4,-20,  4, -1], [-1,  4,  4,  4, -1], [ 0, -1, -1, -1,  0]],
        [[ 1, -2,  2, -2,  1], [-2,  4, -4,  4, -2], [ 2, -4,  4, -4,  2], [-2,  4, -4,  4, -2], [ 1, -2,  2, -2,  1]],
        
        # Edge + diagonal filters (8 filters)
        [[-1,  2, -2,  2, -1], [ 2, -4,  4, -4,  2], [-2,  4, -4,  4, -2], [ 2, -4,  4, -4,  2], [-1,  2, -2,  2, -1]],
        [[ 0,  0,  1,  0,  0], [ 0, -2,  0, -2,  0], [ 1,  0,  4,  0,  1], [ 0, -2,  0, -2,  0], [ 0,  0,  1,  0,  0]],
        [[ 0,  1,  0,  1,  0], [ 1, -2, -4, -2,  1], [ 0, -4, 16, -4,  0], [ 1, -2, -4, -2,  1], [ 0,  1,  0,  1,  0]],
        [[-1,  0,  2,  0, -1], [ 0,  2, -4,  2,  0], [ 2, -4,  4, -4,  2], [ 0,  2, -4,  2,  0], [-1,  0,  2,  0, -1]],
        [[ 1, -2,  2, -2,  1], [-2,  6, -8,  6, -2], [ 2, -8, 12, -8,  2], [-2,  6, -8,  6, -2], [ 1, -2,  2, -2,  1]],
        [[ 0,  0,  1,  0,  0], [ 0, -2,  0, -2,  0], [ 1,  0,  4,  0,  1], [ 0, -2,  0, -2,  0], [ 0,  0,  1,  0,  0]],
        [[ 0, -1,  2, -1,  0], [-1,  4, -6,  4, -1], [ 2, -6,  8, -6,  2], [-1,  4, -6,  4, -1], [ 0, -1,  2, -1,  0]],
        [[-1,  2, -2,  2, -1], [ 2, -6,  8, -6,  2], [-2,  8,-12,  8, -2], [ 2, -6,  8, -6,  2], [-1,  2, -2,  2, -1]],
    ]
    
    return np.array(kernels, dtype=np.float32)


class SRMConv2d(nn.Module):
    """
    Fixed SRM convolutional layer with 30 filters.
    Applies filters to each RGB channel separately (3*30=90 feature maps),
    then reduces to 30 channels via 1x1 convolution.
    """
    
    def __init__(self):
        super(SRMConv2d, self).__init__()
        
        # Get 30 SRM kernels
        srm_kernels = get_srm_kernels()  # (30, 5, 5)
        
        # Create filters for each RGB channel: (30*3, 1, 5, 5)
        # Each of 30 filters is applied to R, G, B separately
        filters = []
        for i in range(3):  # RGB channels
            for kernel in srm_kernels:
                filters.append(kernel)
        filters = np.array(filters, dtype=np.float32)  # (90, 5, 5)
        filters = filters[:, np.newaxis, :, :]  # (90, 1, 5, 5)
        
        # Fixed convolution layer
        self.srm_conv = nn.Conv2d(
            in_channels=3,
            out_channels=90,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=3,
            bias=False
        )
        
        # Set weights and freeze
        self.srm_conv.weight.data = torch.from_numpy(filters)
        self.srm_conv.weight.requires_grad = False
        
        # 1x1 convolution to reduce 90 channels to 30
        self.channel_reduction = nn.Conv2d(90, 30, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Output tensor of shape (B, 30, H, W)
        """
        # Apply SRM filters
        x = self.srm_conv(x)  # (B, 90, H, W)
        
        # Reduce to 30 channels
        x = self.channel_reduction(x)  # (B, 30, H, W)
        
        return x
