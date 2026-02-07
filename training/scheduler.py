"""
Learning rate schedulers for deepfake detection training.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, CosineAnnealingWarmRestarts


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler over N epochs.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # After warmup, keep base learning rate
            return self.base_lrs


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        T_max: Maximum number of iterations for cosine annealing
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            current_epoch = self.last_epoch - self.warmup_epochs
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(current_epoch / self.T_max * math.pi)) / 2
                for base_lr in self.base_lrs
            ]


class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    """
    Cosine annealing with warm restarts and initial warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        T_0: Number of iterations for the first restart
        T_mult: A factor increases T_i after a restart (default: 1)
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.T_cur = 0
        super(CosineAnnealingWarmRestartsWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing with warm restarts
            current_epoch = self.last_epoch - self.warmup_epochs
            
            # Determine current cycle
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
            
            self.T_cur += 1
            
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(self.T_cur / self.T_i * math.pi)) / 2
                for base_lr in self.base_lrs
            ]


def get_scheduler(optimizer, config, mode='visual'):
    """
    Factory function to create learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary containing training parameters
        mode: 'visual' or 'audio' (default: 'visual')
    
    Returns:
        Learning rate scheduler instance
    """
    if mode == 'visual':
        # Visual training has two phases
        # This function returns scheduler for a specific phase
        # Caller should specify which phase in config
        
        if 'phase' not in config or config['phase'] == 1:
            # Phase 1: CosineAnnealing with warmup
            warmup_epochs = config.get('warmup_epochs', 2)
            t_max = config.get('t_max', 8)
            eta_min = config.get('eta_min', 1e-6)
            
            scheduler = CosineAnnealingWithWarmup(
                optimizer,
                warmup_epochs=warmup_epochs,
                T_max=t_max,
                eta_min=eta_min
            )
        else:
            # Phase 2: CosineAnnealingWarmRestarts
            t_0 = config.get('t_0', 4)
            t_mult = config.get('t_mult', 2)
            eta_min = config.get('eta_min', 1e-7)
            warmup_epochs = config.get('warmup_epochs', 0)
            
            scheduler = CosineAnnealingWarmRestartsWithWarmup(
                optimizer,
                warmup_epochs=warmup_epochs,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=eta_min
            )
    
    elif mode == 'audio':
        # Audio training: Single phase with CosineAnnealing
        warmup_epochs = config.get('warmup_epochs', 1)
        t_max = config.get('t_max', 15)
        eta_min = config.get('eta_min', 1e-7)
        
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            T_max=t_max,
            eta_min=eta_min
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'visual' or 'audio'")
    
    return scheduler
