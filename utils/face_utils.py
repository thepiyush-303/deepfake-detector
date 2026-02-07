"""
Face detection and alignment utilities using MTCNN.
"""

import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN


class FaceDetector:
    """
    Face detector wrapper using MTCNN from facenet_pytorch.
    """
    
    def __init__(self, device=None):
        """
        Initialize face detector.
        
        Args:
            device: torch device (default: auto-detect)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False,
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.6]
        )
    
    def detect_faces(self, image, conf_threshold=0.5, min_size=30, margin=0.3):
        """
        Detect faces in an image.
        
        Args:
            image: RGB image as numpy array, shape (H, W, 3)
            conf_threshold: Minimum confidence threshold (default: 0.5)
            min_size: Minimum face size in pixels (default: 30)
            margin: Margin around detected face as fraction of face size (default: 0.3)
        
        Returns:
            Dictionary containing:
                - 'boxes': List of bounding boxes [(x1, y1, x2, y2), ...]
                - 'probs': List of confidence scores
                - 'landmarks': List of facial landmarks (5 points per face)
        """
        if image is None or image.size == 0:
            return {'boxes': [], 'probs': [], 'landmarks': []}
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Detect faces with MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        
        if boxes is None:
            return {'boxes': [], 'probs': [], 'landmarks': []}
        
        # Filter by confidence and size
        valid_faces = []
        valid_probs = []
        valid_landmarks = []
        
        for i, (box, prob, lm) in enumerate(zip(boxes, probs, landmarks)):
            if prob < conf_threshold:
                continue
            
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Check minimum size
            if width < min_size or height < min_size:
                continue
            
            # Apply margin
            margin_x = width * margin
            margin_y = height * margin
            
            x1_margin = max(0, x1 - margin_x)
            y1_margin = max(0, y1 - margin_y)
            x2_margin = min(image.shape[1], x2 + margin_x)
            y2_margin = min(image.shape[0], y2 + margin_y)
            
            valid_faces.append([x1_margin, y1_margin, x2_margin, y2_margin])
            valid_probs.append(prob)
            valid_landmarks.append(lm)
        
        return {
            'boxes': valid_faces,
            'probs': valid_probs,
            'landmarks': valid_landmarks
        }
    
    def align_face(self, image, bbox, landmarks):
        """
        Align face using 5-point facial landmarks.
        
        Args:
            image: RGB image as numpy array, shape (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Facial landmarks array, shape (5, 2) - [left_eye, right_eye, nose, left_mouth, right_mouth]
        
        Returns:
            Aligned face image as numpy array
        """
        if image is None or landmarks is None:
            # Fallback: just crop the bounding box
            x1, y1, x2, y2 = map(int, bbox)
            return image[y1:y2, x1:x2]
        
        # Define reference landmarks (normalized positions for a 112x112 face)
        reference_landmarks = np.array([
            [38.2946, 51.6963],  # Left eye
            [73.5318, 51.5014],  # Right eye
            [56.0252, 71.7366],  # Nose
            [41.5493, 92.3655],  # Left mouth
            [70.7299, 92.2041]   # Right mouth
        ], dtype=np.float32)
        
        # Convert landmarks to numpy array if not already
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks, dtype=np.float32)
        
        # Compute similarity transform
        result = cv2.estimateAffinePartial2D(
            landmarks.reshape(-1, 2),
            reference_landmarks,
            method=cv2.LMEDS
        )
        transform_matrix = result[0] if result is not None else None
        
        if transform_matrix is None:
            # Fallback: just crop the bounding box
            x1, y1, x2, y2 = map(int, bbox)
            return image[y1:y2, x1:x2]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            image,
            transform_matrix,
            (112, 112),
            flags=cv2.INTER_LINEAR
        )
        
        return aligned_face
    
    def detect_and_align_batch(self, images, conf_threshold=0.5, min_size=30, margin=0.3):
        """
        Detect and align faces in a batch of images.
        
        Args:
            images: List of RGB images
            conf_threshold: Minimum confidence threshold
            min_size: Minimum face size in pixels
            margin: Margin around detected face
        
        Returns:
            List of dictionaries, each containing:
                - 'aligned_faces': List of aligned face images
                - 'boxes': List of bounding boxes
                - 'probs': List of confidence scores
        """
        results = []
        
        for image in images:
            detection = self.detect_faces(image, conf_threshold, min_size, margin)
            aligned_faces = []
            
            for bbox, landmarks in zip(detection['boxes'], detection['landmarks']):
                aligned_face = self.align_face(image, bbox, landmarks)
                aligned_faces.append(aligned_face)
            
            results.append({
                'aligned_faces': aligned_faces,
                'boxes': detection['boxes'],
                'probs': detection['probs']
            })
        
        return results
