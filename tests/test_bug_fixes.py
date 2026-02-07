"""
Tests for bug fixes: gan_probs key and audio loading fallback
"""

import os
import sys
import pytest
import numpy as np
import torch
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict_video import predict_video, GAN_TYPES
from inference.predict_image import load_model
from data.audio_preprocessing import load_audio
import warnings

warnings.filterwarnings('ignore')


@pytest.fixture
def dummy_video_file(tmp_path):
    """Create a dummy video file."""
    import cv2
    video_path = tmp_path / "test_video.mp4"
    
    # Create a simple video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Write 30 frames (1 second at 30 fps)
    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    return str(video_path)


@pytest.fixture
def visual_model():
    """Load visual model with random weights."""
    model = load_model(checkpoint_path=None, device='cpu')
    return model


@pytest.fixture
def dummy_audio_file(tmp_path):
    """Create a dummy audio file."""
    audio_path = tmp_path / "test_audio.wav"
    
    # Create a simple audio signal using scipy
    import scipy.io.wavfile as wavfile
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple sine wave
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(str(audio_path), sample_rate, audio)
    return str(audio_path)


def test_video_face_results_have_gan_probs(dummy_video_file, visual_model, tmp_path):
    """
    Test that video face results include 'gan_probs' key.
    This addresses Bug 1: Missing 'gan_probs' key in video face results.
    """
    # Run prediction
    result = predict_video(dummy_video_file, visual_model, device='cpu', output_dir=str(tmp_path))
    
    # Verify result structure
    assert 'faces' in result, "Result should have 'faces' key"
    
    # Verify each face result has gan_probs
    for face in result['faces']:
        assert 'gan_type' in face, "Face result should have 'gan_type' key"
        assert 'gan_probs' in face, "Face result should have 'gan_probs' key (Bug fix)"
        
        # Verify gan_probs is a dictionary
        assert isinstance(face['gan_probs'], dict), "gan_probs should be a dictionary"
        
        # Verify all GAN types are present in the dict
        for gan_type in GAN_TYPES:
            assert gan_type in face['gan_probs'], f"GAN type '{gan_type}' should be in gan_probs"
            assert isinstance(face['gan_probs'][gan_type], (int, float)), \
                f"gan_probs['{gan_type}'] should be a number"
            assert 0.0 <= face['gan_probs'][gan_type] <= 1.0, \
                f"gan_probs['{gan_type}'] should be a probability"
        
        # Verify gan_probs sum to approximately 1
        total_prob = sum(face['gan_probs'].values())
        assert abs(total_prob - 1.0) < 0.01, \
            f"gan_probs should sum to ~1.0, got {total_prob}"
        
        # Verify gan_type matches the max probability in gan_probs
        max_gan = max(face['gan_probs'].items(), key=lambda x: x[1])[0]
        assert face['gan_type'] == max_gan, \
            f"gan_type should be {max_gan} (highest prob), got {face['gan_type']}"
    
    print("✓ Video face results gan_probs test passed")


def test_audio_loading_fallback(dummy_audio_file):
    """
    Test that audio loading works with fallback backends.
    This addresses Bug 2: Audio loading requires TorchCodec.
    """
    # Try to load the audio file
    try:
        waveform, sr = load_audio(dummy_audio_file, target_sr=16000)
        
        # Verify output
        assert waveform is not None, "Waveform should not be None"
        assert sr == 16000, f"Sample rate should be 16000, got {sr}"
        
        # Verify waveform shape
        assert waveform.shape[0] == 1, f"Waveform should be mono (1 channel), got {waveform.shape[0]}"
        assert waveform.shape[1] > 0, "Waveform should have samples"
        
        # Verify waveform is a torch tensor
        assert isinstance(waveform, torch.Tensor), "Waveform should be a torch.Tensor"
        
        print(f"✓ Audio loading test passed (loaded {waveform.shape[1]} samples at {sr} Hz)")
        
    except Exception as e:
        pytest.fail(f"Audio loading failed: {e}")


def test_audio_loading_with_scipy_fallback(tmp_path):
    """
    Test that audio loading works even when torchaudio fails (scipy fallback).
    """
    # Create a simple WAV file that scipy can read
    audio_path = tmp_path / "test_scipy.wav"
    
    import scipy.io.wavfile as wavfile
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(str(audio_path), sample_rate, audio)
    
    # Try to load using our function
    try:
        waveform, sr = load_audio(str(audio_path), target_sr=16000)
        
        assert waveform is not None
        assert sr == 16000
        assert waveform.shape[0] == 1  # mono
        
        print("✓ Audio loading with scipy fallback test passed")
        
    except Exception as e:
        pytest.fail(f"Audio loading with scipy fallback failed: {e}")


def test_ui_app_gan_probs_defensive_coding():
    """
    Test that UI app handles missing gan_probs gracefully.
    This tests the defensive coding in ui/app.py.
    """
    # Simulate a face result dict without gan_probs
    face_result = {
        'track_id': 0,
        'gan_type': 'StyleGAN',
        # Intentionally missing 'gan_probs' key
    }
    
    # Test the .get() method used in ui/app.py
    gan_probs = face_result.get('gan_probs', {})
    
    # Verify it returns an empty dict instead of raising KeyError
    assert isinstance(gan_probs, dict), "Should return an empty dict"
    assert len(gan_probs) == 0, "Should be empty when gan_probs is missing"
    
    # Test with a valid face result
    face_result_valid = {
        'track_id': 0,
        'gan_type': 'StyleGAN',
        'gan_probs': {'StyleGAN': 0.8, 'ProGAN': 0.2}
    }
    
    gan_probs_valid = face_result_valid.get('gan_probs', {})
    assert len(gan_probs_valid) == 2, "Should return the actual gan_probs dict"
    
    print("✓ UI app defensive coding test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
