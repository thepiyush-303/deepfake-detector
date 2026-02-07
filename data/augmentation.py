"""
Forensically-safe data augmentation for deepfake detection.
"""

import torch
import torchvision.transforms as T
import numpy as np
import random
import cv2
from PIL import Image
import io


class VisualAugmentation:
    """
    Forensically-safe augmentations for visual data.
    Includes: HorizontalFlip, RandomCrop, JPEG compression, RandomErasing.
    """
    
    def __init__(self):
        """
        Initialize augmentation pipeline with forensically-safe transforms.
        """
        self.horizontal_flip_prob = 0.5
        self.random_crop_prob = 0.3
        self.jpeg_compression_prob = 0.2
        self.random_erasing_prob = 0.1
    
    def horizontal_flip(self, fingerprints):
        """Apply horizontal flip to all fingerprints."""
        if random.random() < self.horizontal_flip_prob:
            fingerprints['fft'] = np.flip(fingerprints['fft'], axis=1).copy()
            fingerprints['dct'] = np.flip(fingerprints['dct'], axis=1).copy()
            fingerprints['srm'] = np.flip(fingerprints['srm'], axis=2).copy()
            fingerprints['rgb'] = np.flip(fingerprints['rgb'], axis=2).copy()
        return fingerprints
    
    def random_crop(self, fingerprints, crop_size=224, original_size=256):
        """Randomly crop from 256 to 224."""
        if random.random() < self.random_crop_prob:
            max_offset = original_size - crop_size
            offset_h = random.randint(0, max_offset)
            offset_w = random.randint(0, max_offset)
            
            fingerprints['fft'] = fingerprints['fft'][
                offset_h:offset_h+crop_size, offset_w:offset_w+crop_size
            ]
            fingerprints['dct'] = fingerprints['dct'][
                offset_h:offset_h+crop_size, offset_w:offset_w+crop_size
            ]
            fingerprints['srm'] = fingerprints['srm'][
                :, offset_h:offset_h+crop_size, offset_w:offset_w+crop_size
            ]
            fingerprints['rgb'] = fingerprints['rgb'][
                :, offset_h:offset_h+crop_size, offset_w:offset_w+crop_size
            ]
        return fingerprints
    
    def jpeg_compression(self, fingerprints, quality_range=(85, 100)):
        """Apply JPEG compression with quality factor 85-100."""
        if random.random() < self.jpeg_compression_prob:
            # Only apply to RGB (others are derived from it in real use)
            # Convert RGB back to image
            rgb = fingerprints['rgb'].copy()
            rgb = rgb.transpose(1, 2, 0)  # (H, W, 3)
            
            # Denormalize from ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb = rgb * std + mean
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            
            # Apply JPEG compression
            quality = random.randint(*quality_range)
            image_pil = Image.fromarray(rgb)
            buffer = io.BytesIO()
            image_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            image_compressed = Image.open(buffer)
            rgb_compressed = np.array(image_compressed).astype(np.float32) / 255.0
            
            # Re-normalize with ImageNet stats
            rgb_compressed = (rgb_compressed - mean) / std
            fingerprints['rgb'] = rgb_compressed.transpose(2, 0, 1)
        
        return fingerprints
    
    def random_erasing(self, fingerprints, scale=(0.02, 0.1), ratio=(0.3, 3.3)):
        """Apply random erasing to RGB only."""
        if random.random() < self.random_erasing_prob:
            rgb = fingerprints['rgb']
            _, h, w = rgb.shape
            
            area = h * w
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)
            
            erase_h = int(np.sqrt(target_area * aspect_ratio))
            erase_w = int(np.sqrt(target_area / aspect_ratio))
            
            if erase_h < h and erase_w < w:
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)
                
                # Fill with random values
                fingerprints['rgb'][:, i:i+erase_h, j:j+erase_w] = \
                    np.random.randn(3, erase_h, erase_w).astype(np.float32)
        
        return fingerprints
    
    def __call__(self, fingerprints):
        """Apply all augmentations in sequence."""
        fingerprints = self.horizontal_flip(fingerprints)
        fingerprints = self.random_crop(fingerprints)
        fingerprints = self.jpeg_compression(fingerprints)
        fingerprints = self.random_erasing(fingerprints)
        return fingerprints


class AudioAugmentation:
    """
    Forensically-safe augmentations for audio data.
    Includes: Time masking, Random gain.
    """
    
    def __init__(self):
        """
        Initialize augmentation pipeline with forensically-safe transforms.
        """
        self.time_masking_prob = 0.3
        self.random_gain_prob = 0.2
        self.max_time_mask = 20  # Maximum number of time frames to mask
        self.gain_range = (0.8, 1.2)
    
    def time_masking(self, features, max_mask_frames=20):
        """Apply time masking to spectrogram features."""
        if random.random() < self.time_masking_prob:
            # features is a tensor of shape (n_features, time)
            time_steps = features.shape[1]
            mask_frames = random.randint(0, min(max_mask_frames, time_steps // 4))
            
            if mask_frames > 0 and time_steps > mask_frames:
                t0 = random.randint(0, time_steps - mask_frames)
                features[:, t0:t0+mask_frames] = 0
        
        return features
    
    def random_gain(self, features, gain_range=(0.8, 1.2)):
        """Apply random gain to features."""
        if random.random() < self.random_gain_prob:
            gain = random.uniform(*gain_range)
            features = features * gain
        
        return features
    
    def __call__(self, features):
        """Apply all augmentations in sequence."""
        if isinstance(features, torch.Tensor):
            features = features.clone()
        elif isinstance(features, np.ndarray):
            features = features.copy()
            features = torch.from_numpy(features)
        
        features = self.time_masking(features, max_mask_frames=self.max_time_mask)
        features = self.random_gain(features, gain_range=self.gain_range)
        
        return features


class ComposeTransforms:
    """
    Compose multiple transforms together.
    """
    
    def __init__(self, transforms):
        """
        Args:
            transforms: List of transform callables
        """
        self.transforms = transforms
    
    def __call__(self, data):
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            data = transform(data)
        return data
