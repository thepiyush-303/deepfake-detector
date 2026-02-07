import torch
import torch.nn as nn
import timm


class ConstrainedConv2d(nn.Module):
    """
    Convolutional layer with center-weight constraint.
    Constraint: w[i,j,1,1] = -sum(w[i,j]) + w[i,j,1,1]
    This ensures the filter suppresses low-frequency content.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConstrainedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=1, padding=padding, bias=False)
        self.kernel_size = kernel_size
    
    def enforce_constraint(self):
        """
        Enforce the constraint: center weight = -sum(all weights) + center weight
        This makes each filter sum to 0 (high-pass characteristic)
        """
        with torch.no_grad():
            # Get weight tensor: (out_channels, in_channels, kernel_size, kernel_size)
            weight = self.conv.weight.data
            
            # Compute sum of all weights for each filter
            weight_sum = weight.sum(dim=(2, 3), keepdim=True)  # (out_channels, in_channels, 1, 1)
            
            # Get center position
            center = self.kernel_size // 2
            
            # Update center weight: w[i,j,center,center] = -sum(w[i,j]) + w[i,j,center,center]
            weight[:, :, center, center] = -weight_sum.squeeze() + weight[:, :, center, center]
    
    def forward(self, x):
        return self.conv(x)


class NoiseBranch(nn.Module):
    """
    Noise analysis branch using SRM filters and constrained convolution.
    
    Input: 30-channel SRM filter output
    Architecture:
        - Constrained Conv: Conv(30->16) with center-weight constraint + BN + ReLU
        - Channel reduction: Conv(16->3) + BN + ReLU
        - Backbone: EfficientNet-B0 (pretrained)
        - Blocks 0-5: FROZEN
        - Blocks 6-8: UNFROZEN
    Output: 1280-d feature vector
    """
    
    def __init__(self, pretrained=True):
        super(NoiseBranch, self).__init__()
        
        # Constrained convolution layer
        self.constrained_conv = ConstrainedConv2d(30, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Channel reduction to convert to RGB-like format
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, 
                                         features_only=False, num_classes=0)
        
        # Freeze blocks 0-5, unfreeze blocks 6-8
        self._configure_freezing()
    
    def _configure_freezing(self):
        """
        Freeze blocks 0-5 of EfficientNet-B0, unfreeze blocks 6-8.
        """
        # Freeze initial convolution and batch norm
        for param in self.backbone.conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        
        # Freeze blocks 0-5
        for i in range(6):
            if i < len(self.backbone.blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False
        
        # Unfreeze blocks 6-8
        for i in range(6, min(9, len(self.backbone.blocks))):
            if i < len(self.backbone.blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
        
        # Unfreeze final layers
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 30, H, W) - SRM filter output
        
        Returns:
            Feature tensor of shape (B, 1280)
        """
        # Apply constrained convolution
        # Enforce constraint before forward pass
        self.constrained_conv.enforce_constraint()
        
        x = self.constrained_conv(x)  # (B, 16, H, W)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Reduce to 3 channels
        x = self.channel_reduction(x)  # (B, 3, H, W)
        
        # Extract features via EfficientNet-B0
        x = self.backbone(x)  # (B, 1280)
        
        return x
