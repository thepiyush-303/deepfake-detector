"""
Audio deepfake prediction with segmentation and temporal consistency analysis.
"""

import os
import numpy as np
import torch
import warnings

from models.audio_model import AudioDeepfakeModel
from data.audio_preprocessing import load_audio, segment_audio, compute_mel_spectrogram, compute_lfcc


# Vocoder type classes
VOCODER_TYPES = ['WaveNet', 'WaveGlow', 'MelGAN', 'HiFiGAN', 'Parallel WaveGAN', 'Multi-band MelGAN', 'WaveRNN']


def load_audio_model(checkpoint_path=None, device='cuda'):
    """
    Load AudioDeepfakeModel for inference.
    
    Args:
        checkpoint_path: Path to checkpoint file (optional)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = AudioDeepfakeModel(pretrained=True)
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
    
    model.eval()
    return model


def predict_audio(audio_path, model, device='cuda'):
    """
    Predict if audio is real or fake with temporal consistency analysis.
    
    Args:
        audio_path: Path to input audio file
        model: AudioDeepfakeModel instance
        device: Device to run inference on
    
    Returns:
        Dictionary containing:
            - verdict: 'REAL' or 'FAKE'
            - fakeness_score: Overall fakeness score (0-1)
            - realness_score: Overall realness score (0-1)
            - confidence: 'HIGH', 'MEDIUM', or 'LOW'
            - vocoder_type: Most likely vocoder type if fake
            - vocoder_probs: Dictionary of vocoder type probabilities
            - per_segment_scores: List of per-segment fakeness scores
            - consistency: 'STABLE', 'MODERATE', or 'UNSTABLE'
            - temporal_stats: Dictionary with mean, std, median
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading audio from {audio_path}...")
    
    # Load audio
    waveform, sr = load_audio(audio_path, target_sr=16000)
    
    # Segment audio (4s segments with 2s hop)
    print("Segmenting audio...")
    segments = segment_audio(waveform, sr, segment_duration=4.0, hop_duration=2.0, max_segments=30)
    print(f"Created {len(segments)} audio segments")
    
    # Extract features and run inference on each segment
    per_segment_scores = []
    per_segment_vocoder_probs = []
    
    print("Processing segments...")
    for i, segment in enumerate(segments):
        try:
            # Compute mel spectrogram
            mel_spec = compute_mel_spectrogram(segment, sr)
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 80, T)
            
            # Compute LFCC
            lfcc = compute_lfcc(segment, sr)
            lfcc = lfcc.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 40, T)
            
            # Run inference
            with torch.no_grad():
                binary_probs, vocoder_type_probs, _ = model(mel_spec, lfcc, return_probs=True)
            
            fakeness_score = binary_probs.squeeze().cpu().item()
            vocoder_probs = vocoder_type_probs.squeeze().cpu().numpy()
            
            per_segment_scores.append(fakeness_score)
            per_segment_vocoder_probs.append(vocoder_probs)
            
        except Exception as e:
            warnings.warn(f"Error processing segment {i}: {e}")
            continue
    
    if len(per_segment_scores) == 0:
        raise ValueError("No segments could be processed")
    
    # Overlap-aware temporal aggregation
    # Weight segments based on their position (middle segments get higher weight due to more context)
    num_segments = len(per_segment_scores)
    
    if num_segments == 1:
        weights = np.array([1.0])
    else:
        # Create triangular weights favoring middle segments
        weights = np.ones(num_segments)
        
        # For overlapping segments, weight by number of times each time point is covered
        # Since we use 4s segments with 2s hop, most points are covered twice
        # Edge segments have less overlap, so we slightly downweight them
        if num_segments > 2:
            weights[0] = 0.8
            weights[-1] = 0.8
        
        weights = weights / np.sum(weights)
    
    # Weighted aggregation
    scores_array = np.array(per_segment_scores)
    weighted_mean = np.average(scores_array, weights=weights)
    
    # Unweighted statistics for consistency
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    median_score = np.median(scores_array)
    
    # Aggregate vocoder probabilities
    avg_vocoder_probs = np.mean(per_segment_vocoder_probs, axis=0)
    vocoder_probs_dict = {vocoder: float(prob) for vocoder, prob in zip(VOCODER_TYPES, avg_vocoder_probs)}
    
    # Determine most likely vocoder type
    vocoder_type_idx = np.argmax(avg_vocoder_probs)
    vocoder_type = VOCODER_TYPES[vocoder_type_idx]
    
    # Determine consistency based on standard deviation
    if std_score <= 0.10:
        consistency = 'STABLE'
    elif std_score <= 0.25:
        consistency = 'MODERATE'
    else:
        consistency = 'UNSTABLE'
    
    # Overall verdict (use weighted mean)
    overall_score = weighted_mean
    verdict = 'FAKE' if overall_score > 0.5 else 'REAL'
    realness_score = 1.0 - overall_score
    
    # Calculate confidence
    confidence_value = abs(overall_score - 0.5)
    if confidence_value >= 0.3:
        confidence = 'HIGH'
    elif confidence_value >= 0.15:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    return {
        'verdict': verdict,
        'fakeness_score': float(overall_score),
        'realness_score': float(realness_score),
        'confidence': confidence,
        'vocoder_type': vocoder_type,
        'vocoder_probs': vocoder_probs_dict,
        'per_segment_scores': [float(s) for s in per_segment_scores],
        'consistency': consistency,
        'temporal_stats': {
            'mean': float(mean_score),
            'std': float(std_score),
            'median': float(median_score),
            'weighted_mean': float(weighted_mean)
        }
    }
