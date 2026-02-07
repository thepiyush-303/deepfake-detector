# Data Module Documentation

This module provides comprehensive data preprocessing, dataset classes, and augmentation utilities for deepfake detection.

## Overview

The data module consists of four main components:

1. **preprocessing.py** - Image preprocessing and fingerprint extraction
2. **audio_preprocessing.py** - Audio loading and feature extraction
3. **dataset.py** - PyTorch Dataset classes
4. **augmentation.py** - Forensically-safe data augmentation

## Image Preprocessing (`preprocessing.py`)

### Functions

#### `preprocess_image(image_path, target_size=256)`
Load image, detect face, align, and resize.

**Parameters:**
- `image_path` (str): Path to input image
- `target_size` (int): Target size for output (default: 256)

**Returns:**
- Aligned and resized face image as RGB numpy array (H, W, 3)
- Returns None if no face detected

**Example:**
```python
from data.preprocessing import preprocess_image

image = preprocess_image("path/to/image.jpg", target_size=256)
if image is not None:
    print(f"Preprocessed image shape: {image.shape}")
```

#### `extract_fingerprints(image_rgb)`
Extract multiple forensic fingerprints from an RGB image.

**Parameters:**
- `image_rgb` (np.ndarray): RGB image, shape (H, W, 3), values in [0, 255]

**Returns:**
- Dictionary with keys:
  - `'fft'`: FFT spectrum (H, W)
  - `'dct'`: DCT spectrum (H, W)
  - `'srm'`: SRM noise residual (30, H, W)
  - `'rgb'`: ImageNet normalized RGB (3, H, W)

**Example:**
```python
from data.preprocessing import extract_fingerprints

fingerprints = extract_fingerprints(image_rgb)
print(f"FFT shape: {fingerprints['fft'].shape}")
print(f"DCT shape: {fingerprints['dct'].shape}")
print(f"SRM shape: {fingerprints['srm'].shape}")
print(f"RGB shape: {fingerprints['rgb'].shape}")
```

#### `batch_extract_fingerprints(images)`
Extract fingerprints from a batch of images.

**Parameters:**
- `images` (list): List of RGB images as numpy arrays

**Returns:**
- List of dictionaries, each containing fingerprints for one image

## Audio Preprocessing (`audio_preprocessing.py`)

### Functions

#### `load_audio(audio_path, target_sr=16000)`
Load audio file, convert to mono, resample if needed, apply pre-emphasis.

**Parameters:**
- `audio_path` (str): Path to audio file
- `target_sr` (int): Target sample rate (default: 16000)

**Returns:**
- Tuple of (waveform, sample_rate)
  - waveform: torch.Tensor of shape (1, num_samples)
  - sample_rate: int

#### `segment_audio(waveform, sr, segment_duration=4.0, hop_duration=2.0, max_segments=30)`
Segment audio into overlapping windows.

**Parameters:**
- `waveform` (torch.Tensor): Audio waveform tensor of shape (1, num_samples)
- `sr` (int): Sample rate
- `segment_duration` (float): Duration of each segment in seconds (default: 4.0)
- `hop_duration` (float): Hop duration between segments in seconds (default: 2.0)
- `max_segments` (int): Maximum number of segments to return (default: 30)

**Returns:**
- List of audio segments, each of shape (1, segment_samples)

#### `compute_mel_spectrogram(waveform, sr, n_fft=512, hop_length=160, n_mels=80)`
Compute log mel spectrogram with normalization.

**Returns:**
- Log mel spectrogram tensor of shape (n_mels, time)

#### `compute_lfcc(waveform, sr, n_filters=70, n_lfcc=40)`
Compute Linear Frequency Cepstral Coefficients (LFCC) with CMVN.

**Returns:**
- LFCC features tensor of shape (n_lfcc, time)

#### `extract_audio_features(audio_path)`
Extract complete audio features from an audio file.

**Returns:**
- Dictionary with keys:
  - `'mel'`: List of mel spectrograms
  - `'lfcc'`: List of LFCC features
  - `'segments'`: List of audio segment waveforms

## Dataset Classes (`dataset.py`)

### DeepfakeImageDataset

PyTorch Dataset for image-based deepfake detection.

**Parameters:**
- `image_paths` (list): List of image file paths
- `labels` (list): List of binary labels (0: real, 1: fake)
- `gan_types` (list): List of GAN type labels (integers)
- `transform` (callable, optional): Transform to be applied on fingerprints

**Returns:**
- `__getitem__`: (fingerprints_dict, binary_label, gan_type_label)

**Example:**
```python
from data.dataset import DeepfakeImageDataset
from torch.utils.data import DataLoader

dataset = DeepfakeImageDataset(
    image_paths=["img1.jpg", "img2.jpg"],
    labels=[0, 1],
    gan_types=[0, 1],
    transform=None
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### DeepfakeAudioDataset

PyTorch Dataset for audio-based deepfake detection.

**Parameters:**
- `audio_paths` (list): List of audio file paths
- `labels` (list): List of binary labels (0: real, 1: fake)
- `vocoder_types` (list): List of vocoder type labels (integers)
- `transform` (callable, optional): Transform to be applied on features

**Returns:**
- `__getitem__`: (mel, lfcc, binary_label, vocoder_type_label)

## Augmentation (`augmentation.py`)

### VisualAugmentation

Forensically-safe augmentations for visual data.

**Augmentations included:**
- HorizontalFlip (p=0.5)
- RandomCrop (224 from 256, p=0.3)
- JPEG compression (QF 85-100, p=0.2)
- RandomErasing (p=0.1)

**Excluded (not forensically-safe):**
- Blur, noise injection, color jitter, mixup

**Example:**
```python
from data.augmentation import VisualAugmentation

aug = VisualAugmentation()
augmented = aug(fingerprints)
```

### AudioAugmentation

Forensically-safe augmentations for audio data.

**Augmentations included:**
- Time masking (0-20 frames, p=0.3)
- Random gain ([0.8, 1.2], p=0.2)

**Excluded (not forensically-safe):**
- Time stretch, pitch shift, noise injection

**Example:**
```python
from data.augmentation import AudioAugmentation

aug = AudioAugmentation()
augmented = aug(features)
```

## Complete Training Pipeline Example

```python
from data import DeepfakeImageDataset, VisualAugmentation
from torch.utils.data import DataLoader

# Prepare data
train_image_paths = [...]
train_labels = [...]
train_gan_types = [...]

# Create augmentation
train_aug = VisualAugmentation()

# Create datasets
train_dataset = DeepfakeImageDataset(
    image_paths=train_image_paths,
    labels=train_labels,
    gan_types=train_gan_types,
    transform=train_aug
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for fingerprints, labels, gan_types in train_loader:
        # fingerprints is a dict with keys: 'fft', 'dct', 'srm', 'rgb'
        outputs = model(fingerprints)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Dependencies

- torch >= 2.0.0
- torchvision >= 0.15.0
- torchaudio >= 2.0.0
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- scipy >= 1.10.0
- librosa >= 0.10.0
- facenet-pytorch >= 2.5.0
- Pillow >= 9.5.0

## Notes

1. All preprocessing functions handle edge cases gracefully
2. Dataset classes are compatible with PyTorch DataLoader
3. Augmentations are forensically-safe (preserve GAN fingerprints)
4. Face detection uses MTCNN from facenet-pytorch
5. SRM kernels are fixed and non-trainable
6. Audio features use pre-emphasis filter (α=0.97)
7. All functions return torch tensors or numpy arrays with consistent dtypes
