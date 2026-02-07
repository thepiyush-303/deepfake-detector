"""
Audio loading, segmentation, and feature extraction for deepfake audio detection.
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.fftpack import dct


def load_audio(audio_path, target_sr=16000):
    """
    Load audio file, convert to mono, resample if needed, apply pre-emphasis.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        Tuple of (waveform, sample_rate)
        waveform: torch.Tensor of shape (1, num_samples)
        sample_rate: int
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    # Apply pre-emphasis filter (0.97)
    pre_emphasis = 0.97
    waveform_emphasized = torch.cat([
        waveform[:, 0:1],
        waveform[:, 1:] - pre_emphasis * waveform[:, :-1]
    ], dim=1)
    
    return waveform_emphasized, sr


def segment_audio(waveform, sr, segment_duration=4.0, hop_duration=2.0, max_segments=30):
    """
    Segment audio into overlapping windows.
    
    Args:
        waveform: Audio waveform tensor of shape (1, num_samples)
        sr: Sample rate
        segment_duration: Duration of each segment in seconds (default: 4.0)
        hop_duration: Hop duration between segments in seconds (default: 2.0)
        max_segments: Maximum number of segments to return (default: 30)
    
    Returns:
        List of audio segments, each of shape (1, segment_samples)
    """
    segment_samples = int(segment_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    segments = []
    total_samples = waveform.shape[1]
    
    # Generate segments
    start = 0
    while start + segment_samples <= total_samples and len(segments) < max_segments:
        segment = waveform[:, start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    # If no segments or need at least one, take the first segment_duration or pad
    if len(segments) == 0:
        if total_samples >= segment_samples:
            segments.append(waveform[:, :segment_samples])
        else:
            # Pad if audio is too short
            padded = torch.nn.functional.pad(waveform, (0, segment_samples - total_samples))
            segments.append(padded)
    
    return segments


def compute_mel_spectrogram(waveform, sr, n_fft=512, hop_length=160, n_mels=80):
    """
    Compute log mel spectrogram with normalization.
    
    Args:
        waveform: Audio waveform tensor of shape (1, num_samples)
        sr: Sample rate
        n_fft: FFT size (default: 512)
        hop_length: Hop length (default: 160)
        n_mels: Number of mel filters (default: 80)
    
    Returns:
        Log mel spectrogram tensor of shape (n_mels, time)
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=sr // 2,
        power=2.0
    )
    
    # Compute mel spectrogram
    mel_spec = mel_transform(waveform)  # (1, n_mels, time)
    mel_spec = mel_spec.squeeze(0)  # (n_mels, time)
    
    # Convert to log scale
    log_mel = torch.log(mel_spec + 1e-9)
    
    # Normalize (mean=0, std=1)
    mean = log_mel.mean()
    std = log_mel.std()
    log_mel_normalized = (log_mel - mean) / (std + 1e-9)
    
    return log_mel_normalized


def compute_lfcc(waveform, sr, n_filters=70, n_lfcc=40):
    """
    Compute Linear Frequency Cepstral Coefficients (LFCC) with CMVN.
    
    Args:
        waveform: Audio waveform tensor of shape (1, num_samples)
        sr: Sample rate
        n_filters: Number of linear filters (default: 70)
        n_lfcc: Number of LFCC coefficients (default: 40)
    
    Returns:
        LFCC features tensor of shape (n_lfcc, time)
    """
    # Compute STFT
    n_fft = 512
    hop_length = 160
    
    stft_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )
    
    spectrogram = stft_transform(waveform)  # (1, n_fft//2+1, time)
    spectrogram = spectrogram.squeeze(0)  # (n_fft//2+1, time)
    
    # Create linear filterbank
    n_freqs = spectrogram.shape[0]
    linear_filters = torch.zeros(n_filters, n_freqs)
    
    # Define linear spaced center frequencies
    freq_bins = torch.linspace(0, n_freqs - 1, n_filters + 2)
    
    for i in range(n_filters):
        left = int(freq_bins[i])
        center = int(freq_bins[i + 1])
        right = int(freq_bins[i + 2])
        
        # Triangular filter
        for j in range(left, center):
            if center > left:
                linear_filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                linear_filters[i, j] = (right - j) / (right - center)
    
    # Apply filterbank
    linear_spec = torch.matmul(linear_filters, spectrogram)  # (n_filters, time)
    
    # Log compression
    log_linear = torch.log(linear_spec + 1e-9)
    
    # Apply DCT
    log_linear_np = log_linear.numpy()
    lfcc = dct(log_linear_np, axis=0, type=2, norm='ortho')[:n_lfcc, :]
    
    # Apply CMVN (Cepstral Mean and Variance Normalization)
    lfcc_tensor = torch.from_numpy(lfcc).float()
    mean = lfcc_tensor.mean(dim=1, keepdim=True)
    std = lfcc_tensor.std(dim=1, keepdim=True)
    lfcc_normalized = (lfcc_tensor - mean) / (std + 1e-9)
    
    return lfcc_normalized


def extract_audio_features(audio_path):
    """
    Extract complete audio features from an audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with keys:
            - 'mel': List of mel spectrograms, each of shape (n_mels, time)
            - 'lfcc': List of LFCC features, each of shape (n_lfcc, time)
            - 'segments': List of audio segment waveforms
    """
    try:
        # Load audio
        waveform, sr = load_audio(audio_path)
        
        # Segment audio
        segments = segment_audio(waveform, sr)
        
        # Extract features from each segment
        mel_features = []
        lfcc_features = []
        
        for segment in segments:
            # Compute mel spectrogram
            mel_spec = compute_mel_spectrogram(segment, sr)
            mel_features.append(mel_spec)
            
            # Compute LFCC
            lfcc = compute_lfcc(segment, sr)
            lfcc_features.append(lfcc)
        
        return {
            'mel': mel_features,
            'lfcc': lfcc_features,
            'segments': segments
        }
    
    except Exception as e:
        print(f"Error extracting audio features from {audio_path}: {e}")
        return None
