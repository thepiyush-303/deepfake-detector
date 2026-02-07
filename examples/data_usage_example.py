"""
Example usage of data preprocessing and dataset modules.

This script demonstrates how to use the data modules for:
1. Image preprocessing and fingerprint extraction
2. Audio feature extraction
3. Creating datasets
4. Applying augmentations
5. Using with DataLoader
"""

import torch
from torch.utils.data import DataLoader
import numpy as np

# =============================================================================
# Example 1: Image Preprocessing and Fingerprint Extraction
# =============================================================================
print("Example 1: Image Preprocessing and Fingerprint Extraction")
print("-" * 60)

from data.preprocessing import extract_fingerprints

# For demonstration, create a dummy image
dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

# Extract fingerprints
fingerprints = extract_fingerprints(dummy_image)
print(f"FFT spectrum shape: {fingerprints['fft'].shape}")
print(f"DCT spectrum shape: {fingerprints['dct'].shape}")
print(f"SRM noise residual shape: {fingerprints['srm'].shape}")
print(f"RGB normalized shape: {fingerprints['rgb'].shape}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
