import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class MelSpectrogramBranch(nn.Module):
    """
    Mel Spectrogram branch using ResNet-18 backbone.
    
    Input: (1, 80, T) → pad to (1, 128, T) → resize T to 256
    Architecture:
        - Expand to 3 channels via Conv2d(1, 3, 1)
        - ResNet-18: layer1-2 FROZEN, layer3-4 UNFROZEN
    Output: 512-d feature vector
    """
    
    def __init__(self, pretrained=True):
        super(MelSpectrogramBranch, self).__init__()
        
        # Channel expansion: 1 → 3
        self.channel_expand = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        
        # ResNet-18 backbone
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Configure layer freezing: freeze layer1-2, unfreeze layer3-4
        self._configure_freezing()
    
    def _configure_freezing(self):
        """
        Freeze layer1 and layer2, unfreeze layer3 and layer4.
        """
        # Freeze initial convolution and batch norm
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        
        # Freeze layer1 and layer2
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
        
        # Unfreeze layer3 and layer4
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, 80, T) - Mel spectrogram
        
        Returns:
            Feature tensor of shape (B, 512)
        """
        # Pad frequency dimension: 80 → 128
        pad_freq = 128 - x.size(2)
        if pad_freq > 0:
            x = F.pad(x, (0, 0, 0, pad_freq))  # (B, 1, 128, T)
        
        # Resize time dimension to 256
        x = F.interpolate(x, size=(128, 256), mode='bilinear', align_corners=False)  # (B, 1, 128, 256)
        
        # Expand to 3 channels
        x = self.channel_expand(x)  # (B, 3, 128, 256)
        
        # Extract features via ResNet-18
        x = self.backbone(x)  # (B, 512)
        
        return x


class LFCCBranch(nn.Module):
    """
    LFCC branch using ResNet-18 backbone.
    
    Input: (1, 40, T) → pad to (1, 128, T) → resize T to 256
    Architecture:
        - Expand to 3 channels via Conv2d(1, 3, 1)
        - ResNet-18: layer1-2 FROZEN, layer3-4 UNFROZEN
    Output: 512-d feature vector
    """
    
    def __init__(self, pretrained=True):
        super(LFCCBranch, self).__init__()
        
        # Channel expansion: 1 → 3
        self.channel_expand = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        
        # ResNet-18 backbone
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Configure layer freezing: freeze layer1-2, unfreeze layer3-4
        self._configure_freezing()
    
    def _configure_freezing(self):
        """
        Freeze layer1 and layer2, unfreeze layer3 and layer4.
        """
        # Freeze initial convolution and batch norm
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        
        # Freeze layer1 and layer2
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
        
        # Unfreeze layer3 and layer4
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, 40, T) - LFCC
        
        Returns:
            Feature tensor of shape (B, 512)
        """
        # Pad frequency dimension: 40 → 128
        pad_freq = 128 - x.size(2)
        if pad_freq > 0:
            x = F.pad(x, (0, 0, 0, pad_freq))  # (B, 1, 128, T)
        
        # Resize time dimension to 256
        x = F.interpolate(x, size=(128, 256), mode='bilinear', align_corners=False)  # (B, 1, 128, 256)
        
        # Expand to 3 channels
        x = self.channel_expand(x)  # (B, 3, 128, 256)
        
        # Extract features via ResNet-18
        x = self.backbone(x)  # (B, 512)
        
        return x


class AudioDeepfakeModel(nn.Module):
    """
    Dual-branch audio deepfake detection model.
    
    Architecture:
        - Two branches: Mel Spectrogram and LFCC
        - Each branch uses ResNet-18 backbone
        - Audio fusion head processes concatenated features (2 × 512 = 1024-d)
        - Dual-head output: binary classification + vocoder type classification
    
    Input:
        - mel_spec: (B, 1, 80, T) - Mel spectrogram
        - lfcc: (B, 1, 40, T) - Linear Frequency Cepstral Coefficients
    
    Output:
        - binary_logits: (B, 1) - Sigmoid activated, P(fake_audio)
        - vocoder_type_logits: (B, 7) - Softmax activated, vocoder type probabilities
    """
    
    def __init__(self, pretrained=True):
        super(AudioDeepfakeModel, self).__init__()
        
        # Initialize two branches
        self.mel_branch = MelSpectrogramBranch(pretrained=pretrained)
        self.lfcc_branch = LFCCBranch(pretrained=pretrained)
        
        # Audio fusion head
        # Concatenated features: 2 × 512 = 1024-d
        self.fusion_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        # Binary classification head: P(fake_audio)
        # Note: Sigmoid is included as per architecture spec. For training,
        # consider using BCEWithLogitsLoss and removing this activation.
        self.binary_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Vocoder type classification head (7 classes)
        # Note: Softmax is included as per architecture spec. For training,
        # consider using CrossEntropyLoss and removing this activation.
        self.vocoder_type_head = nn.Sequential(
            nn.Linear(64, 7),
            nn.Softmax(dim=1)
        )
    
    def forward(self, mel_spec, lfcc):
        """
        Forward pass through the audio model.
        
        Args:
            mel_spec: (B, 1, 80, T) - Mel spectrogram
            lfcc: (B, 1, 40, T) - LFCC
        
        Returns:
            Tuple of:
                - binary_logits: (B, 1) - P(fake_audio)
                - vocoder_type_logits: (B, 7) - Vocoder type probabilities
        """
        # Extract features from each branch
        mel_features = self.mel_branch(mel_spec)    # (B, 512)
        lfcc_features = self.lfcc_branch(lfcc)      # (B, 512)
        
        # Concatenate features
        fused_features = torch.cat([mel_features, lfcc_features], dim=1)  # (B, 1024)
        
        # Pass through fusion head
        fusion_output = self.fusion_head(fused_features)  # (B, 64)
        
        # Dual-head outputs
        binary_logits = self.binary_head(fusion_output)          # (B, 1)
        vocoder_type_logits = self.vocoder_type_head(fusion_output)  # (B, 7)
        
        return binary_logits, vocoder_type_logits
    
    def to(self, device):
        """Move model to device."""
        super(AudioDeepfakeModel, self).to(device)
        return self
