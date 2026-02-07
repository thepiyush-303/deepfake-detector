"""
Training script for audio deepfake detection model.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.audio_model import AudioDeepfakeModel
from data.dataset import DeepfakeAudioDataset
from training.losses import CombinedLoss
from training.scheduler import get_scheduler
from utils.metrics import compute_eer, compute_auc, compute_ap


class AudioTrainer:
    """Trainer for audio deepfake detection model."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.best_val_eer = float('inf')
        self.patience_counter = 0
        
        # Initialize model
        self.model = AudioDeepfakeModel(pretrained=True).to(device)
        
        # Configure layer freezing (layer1-2 frozen, layer3-4 unfrozen)
        self._configure_freezing()
        
        # Initialize loss
        loss_config = config['loss']
        self.criterion = CombinedLoss(
            alpha=loss_config['alpha'],
            beta=loss_config['beta'],
            gamma=loss_config['gamma'],
            num_classes=config['model']['audio']['num_vocoder_types'],
            feat_dim=config['model']['audio']['hidden_dims'][-1],
            label_smoothing=loss_config['label_smoothing'],
            device=device
        ).to(device)
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Training config
        self.train_config = config['training']['audio']
        
        # Create optimizer
        self.optimizer = self.create_optimizer()
        
        # Create scheduler
        scheduler_config = {
            'warmup_epochs': self.train_config['warmup_epochs'],
            't_max': self.train_config['t_max'],
            'eta_min': self.train_config['eta_min']
        }
        self.scheduler = get_scheduler(self.optimizer, scheduler_config, mode='audio')
    
    def _configure_freezing(self):
        """Freeze layer1-2, unfreeze layer3-4 for both branches."""
        for branch in [self.model.mel_branch, self.model.lfcc_branch]:
            # Freeze conv1 and bn1
            for param in branch.backbone.conv1.parameters():
                param.requires_grad = False
            for param in branch.backbone.bn1.parameters():
                param.requires_grad = False
            
            # Freeze layer1 and layer2
            for param in branch.backbone.layer1.parameters():
                param.requires_grad = False
            for param in branch.backbone.layer2.parameters():
                param.requires_grad = False
            
            # Unfreeze layer3 and layer4
            for param in branch.backbone.layer3.parameters():
                param.requires_grad = True
            for param in branch.backbone.layer4.parameters():
                param.requires_grad = True
            
            # Keep channel expansion trainable
            for param in branch.channel_expand.parameters():
                param.requires_grad = True
    
    def create_optimizer(self):
        """Create optimizer with different learning rates."""
        lr_backbone = self.train_config['lr_backbone']
        lr_head = self.train_config['lr_head']
        
        # Collect parameters
        backbone_params = []
        head_params = []
        
        # Backbone parameters (layer3-4 for both branches)
        for branch in [self.model.mel_branch, self.model.lfcc_branch]:
            backbone_params.extend(branch.backbone.layer3.parameters())
            backbone_params.extend(branch.backbone.layer4.parameters())
            # Include channel expansion
            backbone_params.extend(branch.channel_expand.parameters())
        
        # Head parameters (fusion head + classification heads)
        head_params.extend(self.model.fusion_head.parameters())
        head_params.extend(self.model.binary_head.parameters())
        head_params.extend(self.model.vocoder_type_head.parameters())
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head}
        ], weight_decay=self.train_config['weight_decay'])
        
        return optimizer
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_losses = {'binary': 0, 'type': 0, 'center': 0}
        
        grad_accum_steps = self.train_config['gradient_accumulation']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (mel_features, lfcc_features, binary_labels, vocoder_types) in enumerate(pbar):
            # Handle batch with segments
            # mel_features: (B, num_segments, 1, n_mels, time)
            # lfcc_features: (B, num_segments, 1, n_lfcc, time)
            
            batch_size = mel_features.size(0)
            num_segments = mel_features.size(1)
            
            # Reshape to (B*num_segments, 1, freq, time)
            mel_features = mel_features.view(-1, 1, mel_features.size(3), mel_features.size(4)).to(self.device)
            lfcc_features = lfcc_features.view(-1, 1, lfcc_features.size(3), lfcc_features.size(4)).to(self.device)
            
            # Expand labels to match segments
            binary_labels = binary_labels.unsqueeze(1).expand(-1, num_segments).reshape(-1).to(self.device)
            vocoder_types = vocoder_types.unsqueeze(1).expand(-1, num_segments).reshape(-1).to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                # Model returns raw logits for training
                binary_logits, vocoder_type_logits, fusion_features = self.model(mel_features, lfcc_features)
                
                # Compute loss using fusion features for center loss
                loss_dict = self.criterion(
                    binary_logits,
                    vocoder_type_logits,
                    fusion_features,
                    binary_labels,
                    vocoder_types
                )
                loss = loss_dict['total'] / grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config['gradient_clip'])
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Track losses
            total_loss += loss_dict['total'].item()
            batch_losses['binary'] += loss_dict['binary'].item()
            batch_losses['type'] += loss_dict['type'].item()
            batch_losses['center'] += loss_dict['center'].item()
            
            pbar.set_postfix({
                'loss': total_loss / (i + 1),
                'binary': batch_losses['binary'] / (i + 1),
                'type': batch_losses['type'] / (i + 1)
            })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        all_binary_preds = []
        all_binary_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for mel_features, lfcc_features, binary_labels, vocoder_types in tqdm(val_loader, desc="Validation"):
                # Handle batch with segments
                batch_size = mel_features.size(0)
                num_segments = mel_features.size(1)
                
                # Reshape to (B*num_segments, 1, freq, time)
                mel_features = mel_features.view(-1, 1, mel_features.size(3), mel_features.size(4)).to(self.device)
                lfcc_features = lfcc_features.view(-1, 1, lfcc_features.size(3), lfcc_features.size(4)).to(self.device)
                
                # Expand labels
                binary_labels_expanded = binary_labels.unsqueeze(1).expand(-1, num_segments).reshape(-1).to(self.device)
                vocoder_types_expanded = vocoder_types.unsqueeze(1).expand(-1, num_segments).reshape(-1).to(self.device)
                
                # Forward pass (with probabilities for evaluation)
                binary_logits, vocoder_type_logits, fusion_features = self.model(mel_features, lfcc_features, return_probs=True)
                
                # Binary predictions (already sigmoid activated) - average over segments
                binary_preds = binary_logits.squeeze().view(batch_size, num_segments).mean(dim=1).cpu().numpy()
                
                # Convert back to logits for loss computation
                binary_logits_loss = torch.logit(binary_logits.clamp(1e-7, 1-1e-7))
                vocoder_type_logits_loss = torch.log(vocoder_type_logits.clamp(1e-7, 1.0))
                
                # Compute loss
                loss_dict = self.criterion(
                    binary_logits_loss,
                    vocoder_type_logits_loss,
                    fusion_features,
                    binary_labels_expanded,
                    vocoder_types_expanded
                )
                val_loss += loss_dict['total'].item()
                
                all_binary_preds.extend(binary_preds)
                all_binary_labels.extend(binary_labels.numpy())
        
        # Compute metrics
        val_loss /= len(val_loader)
        val_eer = compute_eer(all_binary_labels, all_binary_preds)
        val_auc = compute_auc(all_binary_labels, all_binary_preds)
        val_ap = compute_ap(all_binary_labels, all_binary_preds)
        
        return val_loss, val_eer, val_auc, val_ap
    
    def train(self, train_loader, val_loader):
        """Train the model."""
        num_epochs = self.train_config['epochs']
        
        print("\n=== Training Audio Model ===")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_eer, val_auc, val_ap = self.validate(val_loader)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, EER: {val_eer:.4f}, AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model based on EER (lower is better)
            if val_eer < self.best_val_eer:
                self.best_val_eer = val_eer
                self.patience_counter = 0
                self.save_checkpoint('best_audio_model.pth', epoch, val_eer, val_auc)
                print(f"Best model saved with EER: {val_eer:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.train_config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation EER: {self.best_val_eer:.4f}")
    
    def save_checkpoint(self, filename, epoch, val_eer, val_auc):
        """Save model checkpoint."""
        checkpoint_dir = self.config['paths']['checkpoints']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_eer': val_eer,
            'val_auc': val_auc,
            'config': self.config
        }
        
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train audio deepfake detection model')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # TODO: Load datasets (placeholder - implement based on data format)
    # For now, create dummy loaders to show structure
    print("Loading datasets...")
    # train_dataset = DeepfakeAudioDataset(train_paths, train_labels, train_vocoder_types)
    # val_dataset = DeepfakeAudioDataset(val_paths, val_labels, val_vocoder_types)
    # train_loader = DataLoader(train_dataset, batch_size=config['training']['audio']['batch_size'], shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=config['training']['audio']['batch_size'], shuffle=False, num_workers=4)
    
    print("Note: Dataset loading not implemented. Please implement data loading based on your data format.")
    print("Expected format: audio_paths, labels (0/1), vocoder_types (0-6)")
    
    # Initialize trainer
    trainer = AudioTrainer(config, device=device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    # trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
