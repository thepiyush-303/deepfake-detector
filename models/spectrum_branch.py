import torch
import torch.nn as nn
import timm


class SpectrumBranch(nn.Module):
    """
    Frequency analysis branch using FFT + DCT magnitude spectrums.
    
    Input: 2 channels (FFT magnitude + DCT magnitude)
    Architecture:
        - Adapter: Conv(2->32->3) with BatchNorm and ReLU
        - Backbone: EfficientNet-B0 (pretrained)
        - Blocks 0-5: FROZEN
        - Blocks 6-8: UNFROZEN
    Output: 1280-d feature vector
    """
    
    def __init__(self, pretrained=True):
        super(SpectrumBranch, self).__init__()
        
        # Adapter to convert 2-channel spectrum input to 3-channel RGB-like format
        self.adapter = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=False, num_classes=0)
        
        # Freeze blocks 0-5, unfreeze blocks 6-8
        self._configure_freezing()
    
    def _configure_freezing(self):
        """
        Freeze blocks 0-5 of EfficientNet-B0, unfreeze blocks 6-8.
        EfficientNet-B0 has blocks accessible via backbone.blocks
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
            x: Input tensor of shape (B, 2, H, W) - FFT and DCT magnitude spectrums
        
        Returns:
            Feature tensor of shape (B, 1280)
        """
        # Convert 2-channel spectrum to 3-channel RGB-like
        x = self.adapter(x)  # (B, 3, H, W)
        
        # Extract features via EfficientNet-B0
        x = self.backbone(x)  # (B, 1280)
        
        return x
