"""
Image deepfake prediction with multi-branch fusion model.
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import warnings

from models.fusion_model import DeepfakeFusionModel
from data.preprocessing import extract_fingerprints, preprocess_image


# GAN type classes
GAN_TYPES = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'BigGAN', 'CycleGAN', 'StarGAN', 'GauGAN']


def load_model(checkpoint_path=None, device='cuda'):
    """
    Load DeepfakeFusionModel for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file (optional)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize model - use pretrained=False if no checkpoint provided to avoid network access
    pretrained = False if checkpoint_path is None else True
    try:
        model = DeepfakeFusionModel(pretrained=pretrained)
    except Exception as e:
        # If pretrained download fails, fall back to random initialization
        warnings.warn(f"Failed to load pretrained weights: {e}. Using random initialization.")
        model = DeepfakeFusionModel(pretrained=False)
    
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        if checkpoint_path is not None:
            warnings.warn(f"Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
        else:
            warnings.warn("No checkpoint provided. Using randomly initialized model.")
    
    # Mark whether a real checkpoint was loaded
    model._checkpoint_loaded = (checkpoint_path is not None and os.path.exists(checkpoint_path))
    
    model.eval()
    return model


def predict_image(image_path, model, device='cuda'):
    """
    Predict if an image is real or fake.
    
    Args:
        image_path: Path to input image
        model: DeepfakeFusionModel instance
        device: Device to run inference on
    
    Returns:
        Dictionary containing:
            - verdict: 'REAL' or 'FAKE'
            - fakeness_score: Probability of being fake (0-1)
            - realness_score: Probability of being real (0-1)
            - confidence: 'HIGH', 'MEDIUM', or 'LOW'
            - gan_type: Most likely GAN type if fake
            - gan_probs: Dictionary of GAN type probabilities
            - branch_contributions: Dictionary of branch contribution scores
            - image: Original aligned face image or full image
            - face_detected: Boolean indicating if a face was detected
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Preprocess image: detect face and align (with fallback to full image)
    result_tuple = preprocess_image(image_path, target_size=256)
    
    # Handle both old (single value) and new (tuple) return formats for compatibility
    if isinstance(result_tuple, tuple):
        aligned_face, face_detected = result_tuple
    else:
        # Old format for compatibility
        aligned_face = result_tuple
        face_detected = aligned_face is not None
    
    if aligned_face is None:
        # Last resort fallback: load image directly and center-crop
        print("⚠️ No face detected, using full image as fallback")
        face_detected = False
        
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Center crop the image to square
        h, w = image_np.shape[:2]
        size = min(h, w)
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        aligned_face = image_np[start_y:start_y+size, start_x:start_x+size]
        
        # Resize to 256x256
        aligned_face = cv2.resize(aligned_face, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    # Extract forensic fingerprints
    fingerprints = extract_fingerprints(aligned_face)
    
    # Prepare inputs
    rgb = torch.from_numpy(fingerprints['rgb']).unsqueeze(0).float().to(device)  # (1, 3, H, W)
    
    # Stack FFT and DCT as spectrum input
    spectrum = np.stack([fingerprints['fft'], fingerprints['dct']], axis=0)  # (2, H, W)
    spectrum = torch.from_numpy(spectrum).unsqueeze(0).float().to(device)  # (1, 2, H, W)
    
    noise = torch.from_numpy(fingerprints['srm']).unsqueeze(0).float().to(device)  # (1, 30, H, W)
    
    # Run inference
    with torch.no_grad():
        binary_probs, gan_type_probs, features_dict = model(
            rgb, spectrum, noise, return_probs=True
        )
    
    # Extract predictions
    fakeness_score = binary_probs.squeeze().cpu().item()
    realness_score = 1.0 - fakeness_score
    
    # Determine verdict
    verdict = 'FAKE' if fakeness_score > 0.5 else 'REAL'
    
    # Calculate confidence based on distance from 0.5
    confidence_value = abs(fakeness_score - 0.5)
    if confidence_value >= 0.3:
        confidence = 'HIGH'
    elif confidence_value >= 0.15:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    # Get GAN type predictions
    gan_probs_array = gan_type_probs.squeeze().cpu().numpy()
    gan_probs = {gan_type: float(prob) for gan_type, prob in zip(GAN_TYPES, gan_probs_array)}
    
    # Determine most likely GAN type
    gan_type_idx = np.argmax(gan_probs_array)
    gan_type = GAN_TYPES[gan_type_idx]
    
    # Compute branch contributions
    contributions = model.compute_branch_contributions(features_dict)
    branch_contributions = {
        'spectrum': float(contributions['spectrum'][0]),
        'noise': float(contributions['noise'][0]),
        'rgb': float(contributions['rgb'][0])
    }
    
    return {
        'verdict': verdict,
        'fakeness_score': float(fakeness_score),
        'realness_score': float(realness_score),
        'confidence': confidence,
        'gan_type': gan_type,
        'gan_probs': gan_probs,
        'branch_contributions': branch_contributions,
        'image': aligned_face,
        'face_detected': face_detected,
        'model_trained': getattr(model, '_checkpoint_loaded', False)
    }


def visualize_results(image, results):
    """
    Create visualization of detection results with fingerprints.
    
    Args:
        image: RGB image as numpy array
        results: Dictionary from predict_image
    
    Returns:
        Matplotlib figure
    """
    # Extract fingerprints for visualization
    fingerprints = extract_fingerprints(image)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # FFT spectrum
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(fingerprints['fft'], cmap='viridis')
    ax2.set_title('FFT Spectrum', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # DCT spectrum
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(fingerprints['dct'], cmap='plasma')
    ax3.set_title('DCT Spectrum', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # SRM noise (show average across channels)
    ax4 = plt.subplot(2, 4, 4)
    srm_avg = np.mean(fingerprints['srm'], axis=0)
    ax4.imshow(srm_avg, cmap='gray')
    ax4.set_title('SRM Noise Pattern', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Verdict and scores
    ax5 = plt.subplot(2, 4, 5)
    ax5.axis('off')
    
    verdict_color = 'red' if results['verdict'] == 'FAKE' else 'green'
    verdict_text = f"Verdict: {results['verdict']}\n\n"
    verdict_text += f"Fakeness: {results['fakeness_score']:.4f}\n"
    verdict_text += f"Realness: {results['realness_score']:.4f}\n"
    verdict_text += f"Confidence: {results['confidence']}\n\n"
    verdict_text += f"GAN Type: {results['gan_type']}"
    
    ax5.text(0.5, 0.5, verdict_text, 
             fontsize=14, fontweight='bold',
             ha='center', va='center',
             color=verdict_color,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Branch contributions pie chart
    ax6 = plt.subplot(2, 4, 6)
    contributions = results['branch_contributions']
    labels = ['Spectrum', 'Noise', 'RGB']
    values = [contributions['spectrum'], contributions['noise'], contributions['rgb']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    ax6.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Branch Contributions', fontsize=12, fontweight='bold')
    
    # GAN type probabilities bar chart
    ax7 = plt.subplot(2, 4, 7)
    gan_types = list(results['gan_probs'].keys())
    gan_probs_values = list(results['gan_probs'].values())
    
    bars = ax7.barh(gan_types, gan_probs_values, color='skyblue')
    
    # Highlight the predicted GAN type
    for i, (gan_type, prob) in enumerate(zip(gan_types, gan_probs_values)):
        if gan_type == results['gan_type']:
            bars[i].set_color('orange')
    
    ax7.set_xlabel('Probability', fontsize=10)
    ax7.set_title('GAN Type Probabilities', fontsize=12, fontweight='bold')
    ax7.set_xlim([0, 1])
    
    # Confidence visualization
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    confidence_colors = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'red'}
    confidence_color = confidence_colors.get(results['confidence'], 'gray')
    
    confidence_text = f"Prediction Confidence:\n\n{results['confidence']}"
    ax8.text(0.5, 0.5, confidence_text,
             fontsize=16, fontweight='bold',
             ha='center', va='center',
             color=confidence_color,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    return fig
