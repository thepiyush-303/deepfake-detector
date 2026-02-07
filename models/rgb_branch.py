import torch
import torch.nn as nn
import timm


class RGBBranch(nn.Module):
    """
    Standard RGB baseline branch.
    
    Input: ImageNet-normalized RGB (3 channels)
    Architecture:
        - Backbone: EfficientNet-B0 (pretrained)
        - Blocks 0-7: FROZEN
        - Block 8: UNFROZEN
    Output: 1280-d feature vector
    """
    
    def __init__(self, pretrained=True):
        super(RGBBranch, self).__init__()
        
        # EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, 
                                         features_only=False, num_classes=0)
        
        # Freeze blocks 0-7, unfreeze only block 8
        self._configure_freezing()
    
    def _configure_freezing(self):
        """
        Freeze blocks 0-7 of EfficientNet-B0, unfreeze only block 8.
        """
        # Freeze initial convolution and batch norm
        for param in self.backbone.conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        
        # Freeze blocks 0-7
        for i in range(8):
            if i < len(self.backbone.blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False
        
        # Unfreeze block 8
        if len(self.backbone.blocks) > 8:
            for param in self.backbone.blocks[8].parameters():
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
            x: Input tensor of shape (B, 3, H, W) - ImageNet-normalized RGB
        
        Returns:
            Feature tensor of shape (B, 1280)
        """
        # Extract features via EfficientNet-B0
        x = self.backbone(x)  # (B, 1280)
        
        return x
