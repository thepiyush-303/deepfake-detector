"""
End-to-End Integration Tests for Deepfake Detection Pipeline

These tests verify the complete pipeline from raw input to model output.
All tests work with randomly initialized models (no trained checkpoints required).
"""

import os
import sys
import pytest
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
import tempfile
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict_image import predict_image, load_model
from inference.predict_video import predict_video
from inference.predict_audio import predict_audio, load_audio_model
from utils.metrics import compute_auc, compute_eer, compute_ap
from data.preprocessing import preprocess_image, extract_fingerprints

warnings.filterwarnings('ignore')


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def dummy_image():
    """Create a dummy RGB image."""
    # Create a random image (224x224x3)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return image


@pytest.fixture
def dummy_image_file(dummy_image, tmp_path):
    """Save dummy image to temporary file."""
    image_path = tmp_path / "test_image.jpg"
    Image.fromarray(dummy_image).save(image_path)
    return str(image_path)


@pytest.fixture
def dummy_video_file(tmp_path):
    """Create a dummy video file."""
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
def dummy_audio_file(tmp_path):
    """Create a dummy audio file."""
    audio_path = tmp_path / "test_audio.wav"
    
    # Create a simple audio signal
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


@pytest.fixture
def visual_model():
    """Load visual model with random weights."""
    model = load_model(checkpoint_path=None, device='cpu')
    return model


@pytest.fixture
def audio_model():
    """Load audio model with random weights."""
    model = load_audio_model(checkpoint_path=None, device='cpu')
    return model


# ============================================================================
# IMAGE PIPELINE TESTS
# ============================================================================

def test_image_pipeline(dummy_image_file, visual_model):
    """Test complete image detection pipeline."""
    # Run prediction
    result = predict_image(dummy_image_file, visual_model, device='cpu')
    
    # Verify result structure
    assert 'verdict' in result
    assert 'fakeness_score' in result
    assert 'realness_score' in result
    assert 'confidence' in result
    assert 'gan_type' in result
    assert 'gan_probs' in result
    assert 'branch_contributions' in result
    assert 'image' in result
    
    # Verify verdict is valid
    assert result['verdict'] in ['REAL', 'FAKE']
    
    # Verify scores are probabilities
    assert 0.0 <= result['fakeness_score'] <= 1.0
    assert 0.0 <= result['realness_score'] <= 1.0
    assert abs(result['fakeness_score'] + result['realness_score'] - 1.0) < 1e-5
    
    # Verify confidence is valid
    assert result['confidence'] in ['HIGH', 'MEDIUM', 'LOW']
    
    # Verify GAN probabilities sum to ~1
    gan_prob_sum = sum(result['gan_probs'].values())
    assert abs(gan_prob_sum - 1.0) < 0.01
    
    # Verify branch contributions sum to ~1
    contrib_sum = sum(result['branch_contributions'].values())
    assert abs(contrib_sum - 1.0) < 0.01
    
    # Verify image shape
    assert result['image'].shape == (256, 256, 3) or result['image'].shape[2] == 3
    
    print("✓ Image pipeline test passed")


def test_image_preprocessing(dummy_image):
    """Test image preprocessing (fingerprint extraction)."""
    # Extract forensic fingerprints
    fingerprints = extract_fingerprints(dummy_image)
    
    # Verify all fingerprints are present
    assert 'rgb' in fingerprints
    assert 'fft' in fingerprints
    assert 'dct' in fingerprints
    assert 'srm' in fingerprints
    
    # Verify shapes
    assert fingerprints['rgb'].shape[0] == 3  # RGB channels
    assert fingerprints['fft'].ndim == 2  # 2D spectrum
    assert fingerprints['dct'].ndim == 2  # 2D spectrum
    assert fingerprints['srm'].shape[0] == 30  # 30 SRM filters
    
    print("✓ Image preprocessing test passed")


# ============================================================================
# VIDEO PIPELINE TESTS
# ============================================================================

def test_video_pipeline(dummy_video_file, visual_model, tmp_path):
    """Test complete video detection pipeline."""
    # Run prediction
    result = predict_video(dummy_video_file, visual_model, device='cpu', output_dir=str(tmp_path))
    
    # Verify result structure
    assert 'faces' in result
    assert 'per_frame_scores' in result
    assert 'overall_verdict' in result
    assert 'overall_score' in result
    assert 'confidence' in result
    assert 'consistency' in result
    assert 'temporal_stats' in result
    assert 'annotated_video_path' in result
    
    # Verify verdict is valid
    assert result['overall_verdict'] in ['REAL', 'FAKE']
    
    # Verify score is probability
    assert 0.0 <= result['overall_score'] <= 1.0
    
    # Verify confidence and consistency
    assert result['confidence'] in ['HIGH', 'MEDIUM', 'LOW']
    assert result['consistency'] in ['STABLE', 'MODERATE', 'UNSTABLE']
    
    # Verify temporal stats
    assert 'mean' in result['temporal_stats']
    assert 'std' in result['temporal_stats']
    assert 'median' in result['temporal_stats']
    
    # Verify per_frame_scores is a list
    assert isinstance(result['per_frame_scores'], list)
    
    # Verify per_frame_scores contains flat numbers, not nested lists
    if len(result['per_frame_scores']) > 0:
        for score in result['per_frame_scores']:
            assert isinstance(score, (int, float, np.number)), f"Expected flat number, got {type(score)}: {score}"
            assert 0.0 <= score <= 1.0, f"Score {score} not in valid range [0, 1]"
    
    print("✓ Video pipeline test passed")


# ============================================================================
# AUDIO PIPELINE TESTS
# ============================================================================

def test_audio_pipeline(dummy_audio_file, audio_model):
    """Test complete audio detection pipeline."""
    # Run prediction
    result = predict_audio(dummy_audio_file, audio_model, device='cpu')
    
    # Verify result structure
    assert 'verdict' in result
    assert 'fakeness_score' in result
    assert 'realness_score' in result
    assert 'confidence' in result
    assert 'vocoder_type' in result
    assert 'vocoder_probs' in result
    assert 'per_segment_scores' in result
    assert 'consistency' in result
    assert 'temporal_stats' in result
    
    # Verify verdict is valid
    assert result['verdict'] in ['REAL', 'FAKE']
    
    # Verify scores are probabilities
    assert 0.0 <= result['fakeness_score'] <= 1.0
    assert 0.0 <= result['realness_score'] <= 1.0
    
    # Verify confidence and consistency
    assert result['confidence'] in ['HIGH', 'MEDIUM', 'LOW']
    assert result['consistency'] in ['STABLE', 'MODERATE', 'UNSTABLE']
    
    # Verify per_segment_scores is a list
    assert isinstance(result['per_segment_scores'], list)
    assert len(result['per_segment_scores']) > 0
    
    # Verify all segment scores are probabilities
    for score in result['per_segment_scores']:
        assert 0.0 <= score <= 1.0
    
    # Verify vocoder probabilities
    vocoder_prob_sum = sum(result['vocoder_probs'].values())
    assert abs(vocoder_prob_sum - 1.0) < 0.01
    
    # Verify temporal stats
    assert 'mean' in result['temporal_stats']
    assert 'std' in result['temporal_stats']
    assert 'median' in result['temporal_stats']
    
    print("✓ Audio pipeline test passed")


# ============================================================================
# GRADIO APP TEST
# ============================================================================

def test_gradio_app_creation():
    """Test that Gradio app can be created without errors."""
    try:
        # Import the app module
        from ui import app
        
        # The app should be defined
        assert hasattr(app, 'app')
        
        # The app should be a Gradio Blocks instance
        import gradio as gr
        assert isinstance(app.app, gr.Blocks)
        
        print("✓ Gradio app creation test passed")
        
    except ImportError as e:
        pytest.skip(f"Could not import ui.app: {e}")
    except Exception as e:
        pytest.fail(f"Error creating Gradio app: {e}")


# ============================================================================
# PREPROCESSING PIPELINE TEST
# ============================================================================

def test_preprocessing_pipeline(tmp_path):
    """Test preprocessing on dummy data."""
    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    input_path = tmp_path / "input.jpg"
    cv2.imwrite(str(input_path), dummy_img)
    
    # Test preprocessing (may fail if no face detected, which is expected)
    try:
        result = preprocess_image(str(input_path), target_size=256)
        
        # Handle new tuple return format (aligned_face, face_detected)
        if isinstance(result, tuple):
            aligned_face, face_detected = result
        else:
            aligned_face = result
            face_detected = aligned_face is not None
        
        if aligned_face is not None:
            # If face was detected or fallback used, verify shape
            assert aligned_face.shape[2] == 3  # RGB
            
            # Test fingerprint extraction
            fingerprints = extract_fingerprints(aligned_face)
            assert 'rgb' in fingerprints
            assert 'fft' in fingerprints
            assert 'dct' in fingerprints
            assert 'srm' in fingerprints
            
            if face_detected:
                msg = "✓ Preprocessing pipeline test passed (face detected)"
            else:
                msg = "✓ Preprocessing pipeline test passed (fallback used)"
            print(msg)
        else:
            # No face detected - expected for random image
            print("✓ Preprocessing pipeline test passed (no face detected, as expected)")
            
    except Exception as e:
        # If error is about no face detection, that's expected
        if "No face" in str(e):
            print("✓ Preprocessing pipeline test passed (no face detected, as expected)")
        else:
            raise


# ============================================================================
# METRICS TEST
# ============================================================================

def test_evaluation_metrics():
    """Test evaluation metrics computation."""
    # Create dummy predictions and ground truth
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.95, 0.25])
    
    # Test AUC
    auc = compute_auc(y_true, y_scores)
    assert 0.0 <= auc <= 1.0
    print(f"  AUC: {auc:.4f}")
    
    # Test EER
    eer = compute_eer(y_true, y_scores)
    assert 0.0 <= eer <= 1.0
    print(f"  EER: {eer:.4f}")
    
    # Test AP
    ap = compute_ap(y_true, y_scores)
    assert 0.0 <= ap <= 1.0
    print(f"  AP: {ap:.4f}")
    
    print("✓ Evaluation metrics test passed")


# ============================================================================
# MODEL LOADING TESTS
# ============================================================================

def test_visual_model_loading():
    """Test visual model can be loaded."""
    model = load_model(checkpoint_path=None, device='cpu')
    assert model is not None
    
    # Verify model is in eval mode
    assert not model.training
    
    # Test forward pass with dummy input
    batch_size = 1
    rgb = torch.randn(batch_size, 3, 224, 224)
    spectrum = torch.randn(batch_size, 2, 224, 224)
    noise = torch.randn(batch_size, 30, 224, 224)
    
    with torch.no_grad():
        output = model(rgb, spectrum, noise, return_probs=True)
    
    # Verify output structure
    assert len(output) == 3  # binary_probs, gan_type_probs, features_dict
    
    print("✓ Visual model loading test passed")


def test_audio_model_loading():
    """Test audio model can be loaded."""
    model = load_audio_model(checkpoint_path=None, device='cpu')
    assert model is not None
    
    # Verify model is in eval mode
    assert not model.training
    
    # Test forward pass with dummy input
    batch_size = 1
    mel = torch.randn(batch_size, 1, 80, 100)  # (B, 1, n_mels, time)
    lfcc = torch.randn(batch_size, 1, 40, 100)  # (B, 1, n_lfcc, time)
    
    with torch.no_grad():
        output = model(mel, lfcc, return_probs=True)
    
    # Verify output structure
    assert len(output) == 3  # binary_probs, vocoder_type_probs, features_dict
    
    print("✓ Audio model loading test passed")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
