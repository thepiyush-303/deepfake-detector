"""
Training script for visual deepfake detection model.
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

from models.fusion_model import DeepfakeFusionModel
from data.dataset import DeepfakeImageDataset
from training.losses import CombinedLoss
from training.scheduler import get_scheduler
from utils.metrics import compute_auc, compute_ap


class VisualTrainer:
    """Trainer for visual deepfake detection model."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.current_phase = 1
        
        # Initialize model
        self.model = DeepfakeFusionModel(pretrained=True).to(device)
        
        # Initialize loss
        loss_config = config['loss']
        self.criterion = CombinedLoss(
            alpha=loss_config['alpha'],
            beta=loss_config['beta'],
            gamma=loss_config['gamma'],
            num_classes=config['model']['visual']['num_gan_types'],
            feat_dim=config['model']['visual']['hidden_dims'][-1],
            label_smoothing=loss_config['label_smoothing'],
            device=device
        ).to(device)
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Training config
        self.train_config = config['training']['visual']
        
    def freeze_all_backbone(self):
        """Freeze all EfficientNet blocks (Phase 1)."""
        for branch in [self.model.spectrum_branch, self.model.noise_branch, self.model.rgb_branch]:
            # Freeze all backbone parameters
            for param in branch.backbone.parameters():
                param.requires_grad = False
            
            # Keep adapters trainable (if they exist)
            if hasattr(branch, 'adapter'):
                for param in branch.adapter.parameters():
                    param.requires_grad = True
            if hasattr(branch, 'constrained_conv'):
                for param in branch.constrained_conv.parameters():
                    param.requires_grad = True
            if hasattr(branch, 'channel_reduction'):
                for param in branch.channel_reduction.parameters():
                    param.requires_grad = True
        
        # Keep fusion head trainable
        for param in self.model.fusion_head.parameters():
            param.requires_grad = True
        for param in self.model.binary_head.parameters():
            param.requires_grad = True
        for param in self.model.gan_type_head.parameters():
            param.requires_grad = True
    
    def unfreeze_partial_backbone(self):
        """Unfreeze specific blocks for Phase 2."""
        # Spectrum and Noise: unfreeze blocks 6-8
        for branch in [self.model.spectrum_branch, self.model.noise_branch]:
            for i in range(6, min(9, len(branch.backbone.blocks))):
                for param in branch.backbone.blocks[i].parameters():
                    param.requires_grad = True
            # Unfreeze head layers
            for param in branch.backbone.conv_head.parameters():
                param.requires_grad = True
            for param in branch.backbone.bn2.parameters():
                param.requires_grad = True
        
        # RGB: unfreeze only block 8
        if len(self.model.rgb_branch.backbone.blocks) > 8:
            for param in self.model.rgb_branch.backbone.blocks[8].parameters():
                param.requires_grad = True
        # Unfreeze head layers
        for param in self.model.rgb_branch.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.model.rgb_branch.backbone.bn2.parameters():
            param.requires_grad = True
    
    def create_optimizer_phase1(self):
        """Create optimizer for Phase 1."""
        phase1_config = self.train_config['phase1']
        lr_adapter = phase1_config['lr_adapter']
        lr_head = phase1_config['lr_head']
        
        # Collect parameters
        adapter_params = []
        head_params = []
        
        # Adapter parameters
        for branch in [self.model.spectrum_branch, self.model.noise_branch]:
            if hasattr(branch, 'adapter'):
                adapter_params.extend(branch.adapter.parameters())
        if hasattr(self.model.noise_branch, 'constrained_conv'):
            adapter_params.extend(self.model.noise_branch.constrained_conv.parameters())
            adapter_params.extend(self.model.noise_branch.bn1.parameters())
            adapter_params.extend(self.model.noise_branch.channel_reduction.parameters())
        
        # Head parameters
        head_params.extend(self.model.fusion_head.parameters())
        head_params.extend(self.model.binary_head.parameters())
        head_params.extend(self.model.gan_type_head.parameters())
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': adapter_params, 'lr': lr_adapter},
            {'params': head_params, 'lr': lr_head}
        ], weight_decay=self.train_config['weight_decay'])
        
        return optimizer
    
    def create_optimizer_phase2(self):
        """Create optimizer for Phase 2."""
        phase2_config = self.train_config['phase2']
        lr_backbone = phase2_config['lr_backbone']
        lr_adapter_head = phase2_config['lr_adapter_head']
        
        # Collect parameters
        backbone_params = []
        adapter_head_params = []
        
        # Backbone parameters (unfrozen blocks)
        for branch in [self.model.spectrum_branch, self.model.noise_branch]:
            for i in range(6, min(9, len(branch.backbone.blocks))):
                backbone_params.extend(branch.backbone.blocks[i].parameters())
            backbone_params.extend(branch.backbone.conv_head.parameters())
            backbone_params.extend(branch.backbone.bn2.parameters())
        
        if len(self.model.rgb_branch.backbone.blocks) > 8:
            backbone_params.extend(self.model.rgb_branch.backbone.blocks[8].parameters())
        backbone_params.extend(self.model.rgb_branch.backbone.conv_head.parameters())
        backbone_params.extend(self.model.rgb_branch.backbone.bn2.parameters())
        
        # Adapter and head parameters
        for branch in [self.model.spectrum_branch, self.model.noise_branch]:
            if hasattr(branch, 'adapter'):
                adapter_head_params.extend(branch.adapter.parameters())
        if hasattr(self.model.noise_branch, 'constrained_conv'):
            adapter_head_params.extend(self.model.noise_branch.constrained_conv.parameters())
            adapter_head_params.extend(self.model.noise_branch.bn1.parameters())
            adapter_head_params.extend(self.model.noise_branch.channel_reduction.parameters())
        
        adapter_head_params.extend(self.model.fusion_head.parameters())
        adapter_head_params.extend(self.model.binary_head.parameters())
        adapter_head_params.extend(self.model.gan_type_head.parameters())
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': adapter_head_params, 'lr': lr_adapter_head}
        ], weight_decay=self.train_config['weight_decay'])
        
        return optimizer
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_losses = {'binary': 0, 'type': 0, 'center': 0}
        
        grad_accum_steps = self.train_config['gradient_accumulation']
        
        pbar = tqdm(train_loader, desc=f"Phase {self.current_phase} Epoch {epoch}")
        for i, (fingerprints, binary_labels, gan_types) in enumerate(pbar):
            # Move to device
            rgb = fingerprints['rgb'].to(self.device)
            spectrum = torch.cat([fingerprints['fft'], fingerprints['dct']], dim=1).to(self.device)
            noise = fingerprints['srm'].to(self.device)
            binary_labels = binary_labels.to(self.device)
            gan_types = gan_types.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                # Model returns raw logits for training
                binary_logits, gan_type_logits, features_dict = self.model(rgb, spectrum, noise)
                
                # Compute loss
                loss_dict = self.criterion(
                    binary_logits, 
                    gan_type_logits,
                    features_dict['fused'],
                    binary_labels,
                    gan_types
                )
                loss = loss_dict['total'] / grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config['gradient_clip'])
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            
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
            for fingerprints, binary_labels, gan_types in tqdm(val_loader, desc="Validation"):
                # Move to device
                rgb = fingerprints['rgb'].to(self.device)
                spectrum = torch.cat([fingerprints['fft'], fingerprints['dct']], dim=1).to(self.device)
                noise = fingerprints['srm'].to(self.device)
                binary_labels_gpu = binary_labels.to(self.device)
                gan_types = gan_types.to(self.device)
                
                # Forward pass (with probabilities for evaluation)
                binary_logits, gan_type_logits, features_dict = self.model(rgb, spectrum, noise, return_probs=True)
                
                # Binary predictions (already sigmoid activated)
                binary_preds = binary_logits.squeeze().cpu().numpy()
                
                # Convert back to logits for loss computation
                binary_logits_loss = torch.logit(binary_logits.clamp(1e-7, 1-1e-7))
                gan_type_logits_loss = torch.log(gan_type_logits.clamp(1e-7, 1.0))
                
                # Compute loss
                loss_dict = self.criterion(
                    binary_logits_loss,
                    gan_type_logits_loss,
                    features_dict['fused'],
                    binary_labels_gpu,
                    gan_types
                )
                val_loss += loss_dict['total'].item()
                
                all_binary_preds.extend(binary_preds if binary_preds.ndim > 0 else [binary_preds.item()])
                all_binary_labels.extend(binary_labels.numpy())
        
        # Compute metrics
        val_loss /= len(val_loader)
        val_auc = compute_auc(all_binary_labels, all_binary_preds)
        val_ap = compute_ap(all_binary_labels, all_binary_preds)
        
        return val_loss, val_auc, val_ap
    
    def train_phase(self, train_loader, val_loader, phase):
        """Train a specific phase."""
        self.current_phase = phase
        
        if phase == 1:
            # Phase 1: Head-only training
            print("\n=== Phase 1: Head-Only Training ===")
            self.freeze_all_backbone()
            optimizer = self.create_optimizer_phase1()
            
            phase_config = self.train_config['phase1']
            num_epochs = phase_config['epochs']
            scheduler_config = {
                'phase': 1,
                'warmup_epochs': phase_config['warmup_epochs'],
                't_max': phase_config['t_max'],
                'eta_min': phase_config['eta_min']
            }
        else:
            # Phase 2: Fine-tuning
            print("\n=== Phase 2: Fine-Tuning ===")
            self.unfreeze_partial_backbone()
            optimizer = self.create_optimizer_phase2()
            
            phase_config = self.train_config['phase2']
            num_epochs = phase_config['epochs']
            scheduler_config = {
                'phase': 2,
                'warmup_epochs': 0,
                't_0': phase_config['t_0'],
                't_mult': phase_config['t_mult'],
                'eta_min': phase_config['eta_min']
            }
        
        scheduler = get_scheduler(optimizer, scheduler_config, mode='visual')
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            val_loss, val_auc, val_ap = self.validate(val_loader)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.save_checkpoint(f'best_model_phase{phase}.pth', epoch, val_auc)
                print(f"Best model saved with AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.train_config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Reset patience for next phase
        self.patience_counter = 0
    
    def save_checkpoint(self, filename, epoch, val_auc):
        """Save model checkpoint."""
        checkpoint_dir = self.config['paths']['checkpoints']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_auc': val_auc,
            'config': self.config
        }
        
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train visual deepfake detection model')
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
    # train_dataset = DeepfakeImageDataset(train_paths, train_labels, train_gan_types)
    # val_dataset = DeepfakeImageDataset(val_paths, val_labels, val_gan_types)
    # train_loader = DataLoader(train_dataset, batch_size=config['training']['visual']['batch_size'], shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=config['training']['visual']['batch_size'], shuffle=False, num_workers=4)
    
    print("Note: Dataset loading not implemented. Please implement data loading based on your data format.")
    print("Expected format: image_paths, labels (0/1), gan_types (0-6)")
    
    # Initialize trainer
    trainer = VisualTrainer(config, device=device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train Phase 1 (epochs 1-8)
    # trainer.train_phase(train_loader, val_loader, phase=1)
    
    # Train Phase 2 (epochs 9-20)
    # trainer.train_phase(train_loader, val_loader, phase=2)
    
    print("\nTraining completed!")
    print(f"Best validation AUC: {trainer.best_val_auc:.4f}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
