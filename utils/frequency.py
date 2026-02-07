"""
Frequency domain analysis utilities for GAN fingerprint detection.
Provides FFT and DCT spectrum computation functions.
"""

import numpy as np
import cv2
from scipy import fftpack


def compute_fft_spectrum(image_rgb, target_size=256):
    """
    Compute FFT spectrum of an RGB image.
    
    Args:
        image_rgb: RGB image with pixel values in [0, 255], shape (H, W, 3)
        target_size: Target size for the output spectrum (default: 256)
    
    Returns:
        FFT magnitude spectrum as float32, shape (target_size, target_size)
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Invalid input image")
    
    # Convert to grayscale
    if len(image_rgb.shape) == 3:
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.astype(np.uint8)
    
    # Resize to target size if needed
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert to float
    gray_float = gray.astype(np.float32)
    
    # Apply 2D FFT
    fft = np.fft.fft2(gray_float)
    
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fft_shifted)
    
    # Apply log scale (add 1 to avoid log(0))
    log_magnitude = np.log(magnitude + 1)
    
    # Normalize to [0, 1]
    if log_magnitude.max() > log_magnitude.min():
        normalized = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())
    else:
        normalized = log_magnitude
    
    return normalized.astype(np.float32)


def compute_dct_spectrum(image_rgb, target_size=256):
    """
    Compute DCT spectrum of an RGB image.
    
    Args:
        image_rgb: RGB image with pixel values in [0, 255], shape (H, W, 3)
        target_size: Target size for the output spectrum (default: 256)
    
    Returns:
        DCT coefficient spectrum as float32, shape (target_size, target_size)
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Invalid input image")
    
    # Convert to grayscale
    if len(image_rgb.shape) == 3:
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.astype(np.uint8)
    
    # Resize to target size if needed
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert to float
    gray_float = gray.astype(np.float32)
    
    # Apply 2D DCT using scipy.fftpack
    dct = fftpack.dct(fftpack.dct(gray_float.T, norm='ortho').T, norm='ortho')
    
    # Take absolute values
    dct_abs = np.abs(dct)
    
    # Apply log scale (add 1 to avoid log(0))
    log_dct = np.log(dct_abs + 1)
    
    # Normalize to [0, 1]
    if log_dct.max() > log_dct.min():
        normalized = (log_dct - log_dct.min()) / (log_dct.max() - log_dct.min())
    else:
        normalized = log_dct
    
    return normalized.astype(np.float32)
