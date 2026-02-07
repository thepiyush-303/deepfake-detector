"""
Face detection, alignment, and fingerprint extraction for deepfake detection.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

from utils.face_utils import FaceDetector
from utils.frequency import compute_fft_spectrum, compute_dct_spectrum
from models.srm_kernels import SRMConv2d


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_image(image_path, target_size=256):
    """
    Load image, detect face, align, and resize.
    Falls back to center-cropped full image if no face detected.
    
    Args:
        image_path: Path to input image
        target_size: Target size for output (default: 256)
    
    Returns:
        Tuple of (aligned_face, face_detected):
        - aligned_face: Aligned and resized face image as RGB numpy array (H, W, 3)
                       or center-cropped full image if no face detected
        - face_detected: Boolean indicating if a face was detected
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Initialize face detector
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detector = FaceDetector(device=device)
        
        # Detect faces
        detection = detector.detect_faces(image_np)
        
        if len(detection['boxes']) == 0:
            # No face detected - use center-cropped full image as fallback
            print("⚠️ No face detected, using full image as fallback")
            
            # Center crop the image to square
            h, w = image_np.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            cropped_image = image_np[start_y:start_y+size, start_x:start_x+size]
            
            # Resize to target size
            if cropped_image.shape[0] != target_size or cropped_image.shape[1] != target_size:
                cropped_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            return cropped_image, False
        
        # Use the first detected face
        bbox = detection['boxes'][0]
        landmarks = detection['landmarks'][0]
        
        # Align face
        aligned_face = detector.align_face(image_np, bbox, landmarks)
        
        # Resize to target size
        if aligned_face.shape[0] != target_size or aligned_face.shape[1] != target_size:
            aligned_face = cv2.resize(aligned_face, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        return aligned_face, True
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, False


def extract_fingerprints(image_rgb):
    """
    Extract multiple forensic fingerprints from an RGB image.
    
    Args:
        image_rgb: RGB image as numpy array, shape (H, W, 3), values in [0, 255]
    
    Returns:
        Dictionary with keys:
            - 'fft': FFT spectrum (H, W)
            - 'dct': DCT spectrum (H, W)
            - 'srm': SRM noise residual (30, H, W)
            - 'rgb': ImageNet normalized RGB (3, H, W)
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Invalid input image")
    
    H, W = image_rgb.shape[:2]
    
    # Extract FFT spectrum
    fft_spectrum = compute_fft_spectrum(image_rgb, target_size=H)
    
    # Extract DCT spectrum
    dct_spectrum = compute_dct_spectrum(image_rgb, target_size=H)
    
    # Extract SRM noise residual
    # Convert to torch tensor (C, H, W) and normalize to [0, 1]
    image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Apply SRM filters
    srm_layer = SRMConv2d()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    srm_layer = srm_layer.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        srm_output = srm_layer(image_tensor)  # (1, 30, H, W)
    
    srm_residual = srm_output.squeeze(0).cpu().numpy()  # (30, H, W)
    
    # Normalize RGB with ImageNet statistics
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_normalized = (image_normalized - IMAGENET_MEAN) / IMAGENET_STD
    image_normalized = image_normalized.transpose(2, 0, 1)  # (3, H, W)
    
    return {
        'fft': fft_spectrum.astype(np.float32),
        'dct': dct_spectrum.astype(np.float32),
        'srm': srm_residual.astype(np.float32),
        'rgb': image_normalized.astype(np.float32)
    }


def batch_extract_fingerprints(images):
    """
    Extract fingerprints from a batch of images.
    
    Args:
        images: List of RGB images as numpy arrays, shape (H, W, 3), values in [0, 255]
    
    Returns:
        List of dictionaries, each containing fingerprints for one image
    """
    results = []
    
    # Initialize SRM layer once for efficiency
    srm_layer = SRMConv2d()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    srm_layer = srm_layer.to(device)
    
    for image_rgb in images:
        if image_rgb is None or image_rgb.size == 0:
            results.append(None)
            continue
        
        try:
            H, W = image_rgb.shape[:2]
            
            # Extract FFT spectrum
            fft_spectrum = compute_fft_spectrum(image_rgb, target_size=H)
            
            # Extract DCT spectrum
            dct_spectrum = compute_dct_spectrum(image_rgb, target_size=H)
            
            # Extract SRM noise residual
            image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                srm_output = srm_layer(image_tensor)  # (1, 30, H, W)
            
            srm_residual = srm_output.squeeze(0).cpu().numpy()  # (30, H, W)
            
            # Normalize RGB with ImageNet statistics
            image_normalized = image_rgb.astype(np.float32) / 255.0
            image_normalized = (image_normalized - IMAGENET_MEAN) / IMAGENET_STD
            image_normalized = image_normalized.transpose(2, 0, 1)  # (3, H, W)
            
            results.append({
                'fft': fft_spectrum.astype(np.float32),
                'dct': dct_spectrum.astype(np.float32),
                'srm': srm_residual.astype(np.float32),
                'rgb': image_normalized.astype(np.float32)
            })
        
        except Exception as e:
            print(f"Error extracting fingerprints: {e}")
            results.append(None)
    
    return results
