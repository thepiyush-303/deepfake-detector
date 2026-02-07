"""
Tests for face detection and alignment utilities.

These tests specifically validate the fix for PyTorch tensor handling
in the FaceDetector class.
"""

import os
import sys
import pytest
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.face_utils import FaceDetector


@pytest.fixture
def face_detector():
    """Create a FaceDetector instance."""
    return FaceDetector(device='cpu')


@pytest.fixture
def dummy_image():
    """Create a dummy RGB image with a simple face-like pattern."""
    # Create a 640x480 RGB image with a face-like pattern
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a simple ellipse to simulate a face
    import cv2
    cv2.ellipse(image, (320, 240), (80, 100), 0, 0, 360, (255, 200, 180), -1)
    # Add eyes
    cv2.circle(image, (290, 220), 10, (0, 0, 0), -1)
    cv2.circle(image, (350, 220), 10, (0, 0, 0), -1)
    # Add mouth
    cv2.ellipse(image, (320, 280), (30, 15), 0, 0, 180, (200, 100, 100), -1)
    return image


@pytest.fixture
def dummy_landmarks_numpy():
    """Create dummy landmarks as numpy array."""
    return np.array([
        [290.0, 220.0],  # Left eye
        [350.0, 220.0],  # Right eye
        [320.0, 250.0],  # Nose
        [300.0, 280.0],  # Left mouth
        [340.0, 280.0]   # Right mouth
    ], dtype=np.float32)


@pytest.fixture
def dummy_landmarks_tensor(dummy_landmarks_numpy):
    """Create dummy landmarks as PyTorch tensor."""
    return torch.from_numpy(dummy_landmarks_numpy)


@pytest.fixture
def dummy_bbox():
    """Create a dummy bounding box."""
    return [240, 140, 400, 340]  # [x1, y1, x2, y2]


def test_align_face_with_numpy_landmarks(face_detector, dummy_image, dummy_bbox, dummy_landmarks_numpy):
    """Test align_face with numpy array landmarks (original behavior)."""
    aligned = face_detector.align_face(dummy_image, dummy_bbox, dummy_landmarks_numpy)
    
    # Should return an aligned face
    assert aligned is not None
    assert isinstance(aligned, np.ndarray)
    # Should be 112x112 as per the alignment spec
    assert aligned.shape == (112, 112, 3)
    print("✓ align_face with numpy landmarks test passed")


def test_align_face_with_pytorch_tensor_landmarks(face_detector, dummy_image, dummy_bbox, dummy_landmarks_tensor):
    """Test align_face with PyTorch tensor landmarks (the fix)."""
    aligned = face_detector.align_face(dummy_image, dummy_bbox, dummy_landmarks_tensor)
    
    # Should handle tensor gracefully and return an aligned face
    assert aligned is not None
    assert isinstance(aligned, np.ndarray)
    # Should be 112x112 as per the alignment spec
    assert aligned.shape == (112, 112, 3)
    print("✓ align_face with PyTorch tensor landmarks test passed")


def test_align_face_with_list_landmarks(face_detector, dummy_image, dummy_bbox):
    """Test align_face with list landmarks."""
    landmarks_list = [
        [290.0, 220.0],
        [350.0, 220.0],
        [320.0, 250.0],
        [300.0, 280.0],
        [340.0, 280.0]
    ]
    aligned = face_detector.align_face(dummy_image, dummy_bbox, landmarks_list)
    
    # Should handle list gracefully and return an aligned face
    assert aligned is not None
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (112, 112, 3)
    print("✓ align_face with list landmarks test passed")


def test_align_face_fallback_with_none_landmarks(face_detector, dummy_image, dummy_bbox):
    """Test align_face fallback when landmarks are None."""
    aligned = face_detector.align_face(dummy_image, dummy_bbox, None)
    
    # Should fallback to bbox crop
    assert aligned is not None
    assert isinstance(aligned, np.ndarray)
    # Should be cropped to bbox dimensions
    x1, y1, x2, y2 = map(int, dummy_bbox)
    expected_height = y2 - y1
    expected_width = x2 - x1
    assert aligned.shape[0] == expected_height
    assert aligned.shape[1] == expected_width
    print("✓ align_face fallback with None landmarks test passed")


def test_align_face_fallback_with_invalid_landmarks(face_detector, dummy_image, dummy_bbox):
    """Test align_face fallback with invalid landmarks that cause estimateAffinePartial2D to fail."""
    # Create invalid landmarks (all zeros) that should cause the transform to fail
    invalid_landmarks = np.zeros((5, 2), dtype=np.float32)
    aligned = face_detector.align_face(dummy_image, dummy_bbox, invalid_landmarks)
    
    # Should fallback to bbox crop
    assert aligned is not None
    assert isinstance(aligned, np.ndarray)
    print("✓ align_face fallback with invalid landmarks test passed")


def test_align_face_bounds_checking(face_detector):
    """Test that align_face properly clips bbox coordinates to image bounds."""
    # Create a small image
    small_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Create bbox that extends beyond image bounds
    out_of_bounds_bbox = [-10, -10, 150, 150]
    
    # Should not crash and should clip coordinates
    aligned = face_detector.align_face(small_image, out_of_bounds_bbox, None)
    
    assert aligned is not None
    assert isinstance(aligned, np.ndarray)
    # Should be clipped to valid region (0 to 100)
    assert aligned.shape[0] <= 100
    assert aligned.shape[1] <= 100
    print("✓ align_face bounds checking test passed")


def test_detect_faces_returns_numpy_arrays(face_detector, dummy_image):
    """Test that detect_faces returns numpy arrays, not tensors."""
    result = face_detector.detect_faces(dummy_image, conf_threshold=0.3)
    
    # Should return a dictionary
    assert isinstance(result, dict)
    assert 'boxes' in result
    assert 'probs' in result
    assert 'landmarks' in result
    
    # All should be lists (not tensors)
    assert isinstance(result['boxes'], list)
    assert isinstance(result['probs'], list)
    assert isinstance(result['landmarks'], list)
    
    # If any faces were detected, verify they are numpy arrays/floats
    if len(result['boxes']) > 0:
        for box in result['boxes']:
            assert isinstance(box, (list, np.ndarray))
        for prob in result['probs']:
            assert isinstance(prob, (float, np.floating))
        for lm in result['landmarks']:
            assert isinstance(lm, np.ndarray)
    
    print("✓ detect_faces returns numpy arrays test passed")


def test_detect_faces_empty_image(face_detector):
    """Test detect_faces with empty/None image."""
    result = face_detector.detect_faces(None)
    
    assert result == {'boxes': [], 'probs': [], 'landmarks': []}
    print("✓ detect_faces empty image test passed")


def test_face_detector_device_cpu(face_detector):
    """Test that FaceDetector can be initialized with CPU device."""
    assert str(face_detector.device) == 'cpu'
    assert face_detector.mtcnn is not None
    print("✓ FaceDetector device CPU test passed")


def test_contiguous_array_conversion():
    """Test that the landmark conversion produces contiguous arrays."""
    # Create a non-contiguous tensor using slicing
    tensor = torch.randn(10, 4)
    non_contiguous = tensor[:, ::2]  # Every other column - creates non-contiguous view
    
    # Verify it's actually non-contiguous
    assert not non_contiguous.is_contiguous()
    
    # Simulate the conversion in align_face
    if hasattr(non_contiguous, 'cpu'):
        landmarks = non_contiguous.cpu().detach().numpy()
    
    landmarks = np.array(landmarks, dtype=np.float32)
    landmarks = np.ascontiguousarray(landmarks.reshape(-1, 2))
    
    # Should be contiguous now
    assert landmarks.flags['C_CONTIGUOUS']
    assert landmarks.dtype == np.float32
    print("✓ Contiguous array conversion test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
