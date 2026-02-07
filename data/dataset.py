"""
PyTorch Dataset classes for deepfake detection.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from data.preprocessing import preprocess_image, extract_fingerprints
from data.audio_preprocessing import extract_audio_features


class DeepfakeImageDataset(Dataset):
    """
    Dataset for image-based deepfake detection.
    """
    
    def __init__(self, image_paths, labels, gan_types, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of binary labels (0: real, 1: fake)
            gan_types: List of GAN type labels (integers)
            transform: Optional transform to be applied on fingerprints
        """
        self.image_paths = image_paths
        self.labels = labels
        self.gan_types = gan_types
        self.transform = transform
        
        assert len(image_paths) == len(labels) == len(gan_types), \
            "All input lists must have the same length"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            fingerprints_dict: Dictionary with keys 'fft', 'dct', 'srm', 'rgb'
            binary_label: Binary label (0 or 1)
            gan_type_label: GAN type label (integer)
        """
        image_path = self.image_paths[idx]
        binary_label = self.labels[idx]
        gan_type_label = self.gan_types[idx]
        
        try:
            # Preprocess image
            image_rgb = preprocess_image(image_path, target_size=256)
            
            if image_rgb is None:
                # If no face detected, load the original image and resize
                image = Image.open(image_path).convert('RGB')
                image_rgb = np.array(image)
                import cv2
                image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # Extract fingerprints
            fingerprints = extract_fingerprints(image_rgb)
            
            # Apply transforms if provided
            if self.transform is not None:
                fingerprints = self.transform(fingerprints)
            
            # Convert to tensors
            fingerprints_tensor = {
                'fft': torch.from_numpy(fingerprints['fft']).float().unsqueeze(0),  # (1, H, W)
                'dct': torch.from_numpy(fingerprints['dct']).float().unsqueeze(0),  # (1, H, W)
                'srm': torch.from_numpy(fingerprints['srm']).float(),  # (30, H, W)
                'rgb': torch.from_numpy(fingerprints['rgb']).float()   # (3, H, W)
            }
            
            return fingerprints_tensor, binary_label, gan_type_label
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data to avoid breaking the batch
            return {
                'fft': torch.zeros(1, 256, 256),
                'dct': torch.zeros(1, 256, 256),
                'srm': torch.zeros(30, 256, 256),
                'rgb': torch.zeros(3, 256, 256)
            }, binary_label, gan_type_label


class DeepfakeAudioDataset(Dataset):
    """
    Dataset for audio-based deepfake detection.
    """
    
    def __init__(self, audio_paths, labels, vocoder_types, transform=None):
        """
        Args:
            audio_paths: List of audio file paths
            labels: List of binary labels (0: real, 1: fake)
            vocoder_types: List of vocoder type labels (integers)
            transform: Optional transform to be applied on features
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.vocoder_types = vocoder_types
        self.transform = transform
        
        assert len(audio_paths) == len(labels) == len(vocoder_types), \
            "All input lists must have the same length"
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            mel: Mel spectrogram tensor of shape (num_segments, n_mels, time)
            lfcc: LFCC features tensor of shape (num_segments, n_lfcc, time)
            binary_label: Binary label (0 or 1)
            vocoder_type_label: Vocoder type label (integer)
        """
        audio_path = self.audio_paths[idx]
        binary_label = self.labels[idx]
        vocoder_type_label = self.vocoder_types[idx]
        
        try:
            # Extract audio features
            features = extract_audio_features(audio_path)
            
            if features is None:
                raise ValueError(f"Failed to extract features from {audio_path}")
            
            mel_features = features['mel']
            lfcc_features = features['lfcc']
            
            # Apply transforms if provided
            if self.transform is not None:
                mel_features = [self.transform(mel) for mel in mel_features]
                lfcc_features = [self.transform(lfcc) for lfcc in lfcc_features]
            
            # Stack segments into tensors
            mel_tensor = torch.stack(mel_features, dim=0)  # (num_segments, n_mels, time)
            lfcc_tensor = torch.stack(lfcc_features, dim=0)  # (num_segments, n_lfcc, time)
            
            return mel_tensor, lfcc_tensor, binary_label, vocoder_type_label
        
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return dummy data to avoid breaking the batch
            # Use common shapes: mel (80, time), lfcc (40, time)
            return torch.zeros(1, 80, 100), torch.zeros(1, 40, 100), binary_label, vocoder_type_label
