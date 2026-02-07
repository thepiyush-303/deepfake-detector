"""
Data module for deepfake detection.
"""

from data.preprocessing import preprocess_image, extract_fingerprints, batch_extract_fingerprints
from data.audio_preprocessing import (
    load_audio,
    segment_audio,
    compute_mel_spectrogram,
    compute_lfcc,
    extract_audio_features
)
from data.dataset import DeepfakeImageDataset, DeepfakeAudioDataset
from data.augmentation import VisualAugmentation, AudioAugmentation, ComposeTransforms

__all__ = [
    'preprocess_image',
    'extract_fingerprints',
    'batch_extract_fingerprints',
    'load_audio',
    'segment_audio',
    'compute_mel_spectrogram',
    'compute_lfcc',
    'extract_audio_features',
    'DeepfakeImageDataset',
    'DeepfakeAudioDataset',
    'VisualAugmentation',
    'AudioAugmentation',
    'ComposeTransforms'
]
