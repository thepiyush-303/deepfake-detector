"""
Loss functions for deepfake detection training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedBCELoss(nn.Module):
    """
    Label smoothed Binary Cross Entropy Loss.
    
    Formula: y_smooth = y * (1 - epsilon) + (1 - y) * epsilon / 2
    
    Args:
        epsilon: Label smoothing factor (default: 0.1)
    """
    
    def __init__(self, epsilon=0.1):
        super(LabelSmoothedBCELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, 1), raw logits (not sigmoid activated)
            targets: Ground truth labels (B,) or (B, 1), values in {0, 1}
        
        Returns:
            Scalar loss value
        """
        # Ensure targets are the correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        targets = targets.float()
        
        # Apply label smoothing: 0 -> epsilon, 1 -> (1 - epsilon)
        targets_smooth = targets * (1 - self.epsilon) + self.epsilon / 2
        
        # Use BCEWithLogitsLoss for numerical stability
        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth)
        
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss with automatic class weight computation.
    
    Class weights = N_total / (N_classes × N_samples_i)
    
    Args:
        num_classes: Number of classes
    """
    
    def __init__(self, num_classes):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = None
    
    def compute_class_weights(self, targets, device):
        """
        Compute class weights based on target distribution.
        
        Args:
            targets: Ground truth labels (B,)
            device: Device to place weights on
        
        Returns:
            Class weights tensor of shape (num_classes,)
        """
        # Count samples per class
        class_counts = torch.bincount(targets, minlength=self.num_classes).float()
        
        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1.0)
        
        # Compute weights: N_total / (N_classes × N_samples_i)
        total_samples = targets.size(0)
        weights = total_samples / (self.num_classes * class_counts)
        
        return weights.to(device)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, num_classes), raw logits
            targets: Ground truth labels (B,), class indices
        
        Returns:
            Scalar loss value
        """
        # Compute class weights from current batch
        device = logits.device
        weights = self.compute_class_weights(targets, device)
        
        # Compute weighted cross entropy loss
        loss = F.cross_entropy(logits, targets, weight=weights)
        
        return loss


class CenterLoss(nn.Module):
    """
    Center Loss for learning discriminative features.
    
    Formula: L = 0.5 × Σ ‖f_i - c_yi‖²
    
    Maintains running mean centers for each class.
    
    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension
        device: Device for center storage
    """
    
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Initialize centers with zeros
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.centers.requires_grad = True
    
    def forward(self, features, targets):
        """
        Args:
            features: Feature vectors (B, feat_dim)
            targets: Ground truth labels (B,), class indices
        
        Returns:
            Scalar center loss value
        """
        batch_size = features.size(0)
        
        # Get centers for the batch
        centers_batch = self.centers.index_select(0, targets.long())  # (B, feat_dim)
        
        # Compute L2 distance: ‖f_i - c_yi‖²
        diff = features - centers_batch
        loss = 0.5 * torch.sum(torch.pow(diff, 2)) / batch_size
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for deepfake detection.
    
    Formula: L_total = α·L_binary + β·L_type + γ·L_center
    
    Args:
        alpha: Weight for binary classification loss (default: 1.0)
        beta: Weight for type classification loss (default: 0.5)
        gamma: Weight for center loss (default: 0.01)
        num_classes: Number of type classes (default: 7)
        feat_dim: Feature dimension for center loss (default: 128)
        label_smoothing: Label smoothing epsilon (default: 0.1)
        device: Device for computation (default: 'cuda')
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.01, num_classes=7, 
                 feat_dim=128, label_smoothing=0.1, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize component losses
        self.binary_loss = LabelSmoothedBCELoss(epsilon=label_smoothing)
        self.type_loss = WeightedCrossEntropyLoss(num_classes=num_classes)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)
    
    def forward(self, binary_logits, type_logits, features, binary_targets, type_targets):
        """
        Args:
            binary_logits: Binary classification logits (B, 1)
            type_logits: Type classification logits (B, num_classes)
            features: Feature vectors for center loss (B, feat_dim)
            binary_targets: Binary labels (B,) or (B, 1)
            type_targets: Type labels (B,)
        
        Returns:
            Dictionary containing:
                - 'total': Total combined loss
                - 'binary': Binary classification loss
                - 'type': Type classification loss
                - 'center': Center loss
        """
        # Compute component losses
        loss_binary = self.binary_loss(binary_logits, binary_targets)
        loss_type = self.type_loss(type_logits, type_targets)
        loss_center = self.center_loss(features, type_targets)
        
        # Combine losses
        total_loss = self.alpha * loss_binary + self.beta * loss_type + self.gamma * loss_center
        
        return {
            'total': total_loss,
            'binary': loss_binary,
            'type': loss_type,
            'center': loss_center
        }
    
    def to(self, device):
        """Move loss to device."""
        super(CombinedLoss, self).to(device)
        self.center_loss.centers = self.center_loss.centers.to(device)
        self.center_loss.device = device
        return self
