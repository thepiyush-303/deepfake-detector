import torch
import torch.nn as nn
from .spectrum_branch import SpectrumBranch
from .noise_branch import NoiseBranch
from .rgb_branch import RGBBranch


class DeepfakeFusionModel(nn.Module):
    """
    Multi-branch fusion network for image/video deepfake detection.
    
    Architecture:
        - Three branches: SpectrumBranch, NoiseBranch, RGBBranch
        - Each branch outputs 1280-d features
        - Fusion head processes concatenated features (3 × 1280 = 3840-d)
        - Dual-head output: binary classification + GAN type classification
    
    Input:
        - rgb: (B, 3, H, W) - ImageNet-normalized RGB
        - spectrum: (B, 2, H, W) - FFT and DCT magnitude spectrums
        - noise: (B, 30, H, W) - SRM filter output
    
    Output:
        - binary_logits: (B, 1) - Sigmoid activated, P(fake)
        - gan_type_logits: (B, 7) - Softmax activated, GAN type probabilities
        - features_dict: Dictionary containing branch-wise features
    """
    
    def __init__(self, pretrained=True):
        super(DeepfakeFusionModel, self).__init__()
        
        # Initialize three branches
        self.spectrum_branch = SpectrumBranch(pretrained=pretrained)
        self.noise_branch = NoiseBranch(pretrained=pretrained)
        self.rgb_branch = RGBBranch(pretrained=pretrained)
        
        # Fusion head
        # Concatenated features: 3 × 1280 = 3840-d
        self.fusion_head = nn.Sequential(
            nn.Linear(3840, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Binary classification head: P(fake)
        self.binary_head = nn.Linear(128, 1)
        
        # GAN type classification head (7 classes)
        self.gan_type_head = nn.Linear(128, 7)
    
    def forward(self, rgb, spectrum, noise, return_probs=False):
        """
        Forward pass through the fusion model.
        
        Args:
            rgb: (B, 3, H, W) - ImageNet-normalized RGB
            spectrum: (B, 2, H, W) - FFT and DCT magnitude spectrums
            noise: (B, 30, H, W) - SRM filter output
            return_probs: If True, apply sigmoid/softmax to outputs (default: False for training)
        
        Returns:
            Tuple of:
                - binary_logits: (B, 1) - Raw logits or P(fake) if return_probs=True
                - gan_type_logits: (B, 7) - Raw logits or probabilities if return_probs=True
                - features_dict: Dictionary with branch features
        """
        # Extract features from each branch
        spectrum_features = self.spectrum_branch(spectrum)  # (B, 1280)
        noise_features = self.noise_branch(noise)            # (B, 1280)
        rgb_features = self.rgb_branch(rgb)                  # (B, 1280)
        
        # Concatenate features
        fused_features = torch.cat([spectrum_features, noise_features, rgb_features], dim=1)  # (B, 3840)
        
        # Pass through fusion head
        fusion_output = self.fusion_head(fused_features)  # (B, 128)
        
        # Dual-head outputs (raw logits)
        binary_logits = self.binary_head(fusion_output)      # (B, 1)
        gan_type_logits = self.gan_type_head(fusion_output)  # (B, 7)
        
        # Apply activations if requested (for inference)
        if return_probs:
            binary_logits = torch.sigmoid(binary_logits)
            gan_type_logits = torch.softmax(gan_type_logits, dim=1)
        
        # Store features for explainability
        features_dict = {
            'spectrum': spectrum_features,
            'noise': noise_features,
            'rgb': rgb_features,
            'fused': fusion_output
        }
        
        return binary_logits, gan_type_logits, features_dict
    
    def compute_branch_contributions(self, features_dict):
        """
        Compute the contribution of each branch for explainability.
        
        Args:
            features_dict: Dictionary containing branch-wise features
        
        Returns:
            Dictionary with normalized contribution scores
        """
        # Extract branch features
        spectrum_features = features_dict['spectrum']  # (B, 1280)
        noise_features = features_dict['noise']        # (B, 1280)
        rgb_features = features_dict['rgb']            # (B, 1280)
        
        # Compute L2 norms as contribution scores
        spectrum_norm = torch.norm(spectrum_features, p=2, dim=1)  # (B,)
        noise_norm = torch.norm(noise_features, p=2, dim=1)        # (B,)
        rgb_norm = torch.norm(rgb_features, p=2, dim=1)            # (B,)
        
        # Normalize to sum to 1
        total_norm = spectrum_norm + noise_norm + rgb_norm + 1e-8  # (B,)
        
        contributions = {
            'spectrum': (spectrum_norm / total_norm).cpu().numpy(),
            'noise': (noise_norm / total_norm).cpu().numpy(),
            'rgb': (rgb_norm / total_norm).cpu().numpy()
        }
        
        return contributions
    
    def to(self, device):
        """Move model to device."""
        super(DeepfakeFusionModel, self).to(device)
        return self
