"""
Video deepfake prediction with face tracking and temporal aggregation.
"""

import os
import numpy as np
import torch
import cv2
from collections import defaultdict
import warnings

from models.fusion_model import DeepfakeFusionModel
from utils.video_utils import extract_frames, track_faces, render_face_overlay, save_annotated_video
from utils.face_utils import FaceDetector
from data.preprocessing import extract_fingerprints


# GAN type classes
GAN_TYPES = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'BigGAN', 'CycleGAN', 'StarGAN', 'GauGAN']


def predict_video(video_path, model, device='cuda', output_dir='output'):
    """
    Predict if a video is real or fake with face tracking and temporal analysis.
    
    Args:
        video_path: Path to input video
        model: DeepfakeFusionModel instance
        device: Device to run inference on
        output_dir: Directory to save annotated video
    
    Returns:
        Dictionary containing:
            - faces: List of tracked faces with per-face results
            - per_frame_scores: List of frame-level scores
            - overall_verdict: 'REAL' or 'FAKE'
            - overall_score: Overall fakeness score
            - confidence: 'HIGH', 'MEDIUM', or 'LOW'
            - consistency: 'STABLE', 'MODERATE', or 'UNSTABLE'
            - temporal_stats: Dictionary with mean, std, median
            - annotated_video_path: Path to saved annotated video
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames at 1 FPS
    print("Extracting frames from video...")
    frames = extract_frames(video_path, target_fps=1, min_frames=8, max_frames=32)
    print(f"Extracted {len(frames)} frames")
    
    # Initialize face detector
    face_detector = FaceDetector(device=device)
    
    # Detect faces in all frames
    print("Detecting faces...")
    all_detections = []
    all_bboxes = []
    
    for frame in frames:
        detection = face_detector.detect_faces(frame, conf_threshold=0.5, min_size=30, margin=0.2)
        all_detections.append(detection)
        all_bboxes.append(detection['boxes'])
    
    # Track faces across frames
    print("Tracking faces...")
    track_ids = track_faces(frames, all_bboxes)
    
    # Organize detections by track
    tracks = defaultdict(list)
    
    for frame_idx, (detection, frame_track_ids) in enumerate(zip(all_detections, track_ids)):
        for bbox, landmarks, track_id in zip(detection['boxes'], detection['landmarks'], frame_track_ids):
            tracks[track_id].append({
                'frame_idx': frame_idx,
                'bbox': bbox,
                'landmarks': landmarks,
                'frame': frames[frame_idx]
            })
    
    print(f"Found {len(tracks)} unique face tracks")
    
    # Process each track
    face_results = []
    per_frame_scores = [[] for _ in range(len(frames))]
    
    for track_id, track_detections in tracks.items():
        print(f"Processing track {track_id} ({len(track_detections)} detections)...")
        
        track_scores = []
        track_gan_probs = []
        
        for detection_info in track_detections:
            frame = detection_info['frame']
            bbox = detection_info['bbox']
            landmarks = detection_info['landmarks']
            frame_idx = detection_info['frame_idx']
            
            try:
                # Crop and align face
                aligned_face = face_detector.align_face(frame, bbox, landmarks)
                
                # Resize to 256x256
                aligned_face = cv2.resize(aligned_face, (256, 256), interpolation=cv2.INTER_LINEAR)
                
                # Extract fingerprints
                fingerprints = extract_fingerprints(aligned_face)
                
                # Prepare inputs
                rgb = torch.from_numpy(fingerprints['rgb']).unsqueeze(0).float().to(device)
                spectrum = np.stack([fingerprints['fft'], fingerprints['dct']], axis=0)
                spectrum = torch.from_numpy(spectrum).unsqueeze(0).float().to(device)
                noise = torch.from_numpy(fingerprints['srm']).unsqueeze(0).float().to(device)
                
                # Run inference
                with torch.no_grad():
                    binary_probs, gan_type_probs, _ = model(rgb, spectrum, noise, return_probs=True)
                
                fakeness_score = binary_probs.squeeze().cpu().item()
                gan_probs = gan_type_probs.squeeze().cpu().numpy()
                
                track_scores.append(fakeness_score)
                track_gan_probs.append(gan_probs)
                
                # Store per-frame score
                per_frame_scores[frame_idx].append(fakeness_score)
                
            except Exception as e:
                warnings.warn(f"Error processing detection in track {track_id}, frame {frame_idx}: {e}")
                continue
        
        if len(track_scores) == 0:
            continue
        
        # Temporal aggregation for this track
        track_scores_array = np.array(track_scores)
        mean_score = np.mean(track_scores_array)
        std_score = np.std(track_scores_array)
        median_score = np.median(track_scores_array)
        
        # Aggregate GAN probabilities
        if len(track_gan_probs) > 0:
            try:
                avg_gan_probs = np.mean(np.stack(track_gan_probs), axis=0)
            except ValueError:
                avg_gan_probs = track_gan_probs[0]  # fallback to first
        else:
            avg_gan_probs = np.zeros(len(GAN_TYPES))
        gan_type_idx = np.argmax(avg_gan_probs)
        gan_type = GAN_TYPES[gan_type_idx]
        
        # Determine consistency for this track
        if std_score <= 0.10:
            track_consistency = 'STABLE'
        elif std_score <= 0.25:
            track_consistency = 'MODERATE'
        else:
            track_consistency = 'UNSTABLE'
        
        face_results.append({
            'track_id': track_id,
            'num_detections': len(track_scores),
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'median_score': float(median_score),
            'verdict': 'FAKE' if mean_score > 0.5 else 'REAL',
            'consistency': track_consistency,
            'gan_type': gan_type,
            'detections': track_detections
        })
    
    if len(face_results) == 0:
        # No faces detected - fall back to analyzing full frames
        print("⚠️ No faces detected in video, using full frames as fallback")
        
        # Process up to 8 frames with center crops
        num_frames_to_process = min(8, len(frames))
        frame_scores = []
        frame_gan_probs = []
        
        for i in range(num_frames_to_process):
            frame = frames[i]
            
            try:
                # Center crop the frame to square
                h, w = frame.shape[:2]
                size = min(h, w)
                start_y = (h - size) // 2
                start_x = (w - size) // 2
                cropped_frame = frame[start_y:start_y+size, start_x:start_x+size]
                
                # Resize to 256x256
                cropped_frame = cv2.resize(cropped_frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                
                # Extract fingerprints
                fingerprints = extract_fingerprints(cropped_frame)
                
                # Prepare inputs
                rgb = torch.from_numpy(fingerprints['rgb']).unsqueeze(0).float().to(device)
                spectrum = np.stack([fingerprints['fft'], fingerprints['dct']], axis=0)
                spectrum = torch.from_numpy(spectrum).unsqueeze(0).float().to(device)
                noise = torch.from_numpy(fingerprints['srm']).unsqueeze(0).float().to(device)
                
                # Run inference
                with torch.no_grad():
                    binary_probs, gan_type_probs, _ = model(rgb, spectrum, noise, return_probs=True)
                
                fakeness_score = binary_probs.squeeze().cpu().item()
                gan_probs = gan_type_probs.squeeze().cpu().numpy()
                
                frame_scores.append(fakeness_score)
                frame_gan_probs.append(gan_probs)
                per_frame_scores[i].append(fakeness_score)
                
            except Exception as e:
                warnings.warn(f"Error processing frame {i}: {e}")
                continue
        
        if len(frame_scores) == 0:
            # Still couldn't process any frames
            raise ValueError("Could not process any frames from the video")
        
        # Create a single "face" result for the full-frame analysis
        mean_score = np.mean(frame_scores)
        std_score = np.std(frame_scores)
        median_score = np.median(frame_scores)
        
        if len(frame_gan_probs) > 0:
            try:
                avg_gan_probs = np.mean(np.stack(frame_gan_probs), axis=0)
            except ValueError:
                avg_gan_probs = frame_gan_probs[0]  # fallback to first
        else:
            avg_gan_probs = np.zeros(len(GAN_TYPES))
        
        gan_type_idx = np.argmax(avg_gan_probs)
        gan_type = GAN_TYPES[gan_type_idx]
        
        if std_score <= 0.10:
            consistency = 'STABLE'
        elif std_score <= 0.25:
            consistency = 'MODERATE'
        else:
            consistency = 'UNSTABLE'
        
        face_results.append({
            'track_id': 0,
            'num_detections': len(frame_scores),
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'median_score': float(median_score),
            'verdict': 'FAKE' if mean_score > 0.5 else 'REAL',
            'consistency': consistency,
            'gan_type': gan_type,
            'detections': []  # No face detections for full-frame analysis
        })
    
    # Overall temporal aggregation
    all_scores = [result['mean_score'] for result in face_results]
    overall_mean = np.mean(all_scores)
    overall_std = np.std(all_scores)
    overall_median = np.median(all_scores)
    
    # Determine overall consistency
    if overall_std <= 0.10:
        overall_consistency = 'STABLE'
    elif overall_std <= 0.25:
        overall_consistency = 'MODERATE'
    else:
        overall_consistency = 'UNSTABLE'
    
    # Overall verdict
    overall_verdict = 'FAKE' if overall_mean > 0.5 else 'REAL'
    
    # Calculate confidence
    confidence_value = abs(overall_mean - 0.5)
    if confidence_value >= 0.3:
        confidence = 'HIGH'
    elif confidence_value >= 0.15:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    # Render annotated video
    print("Rendering annotated video...")
    annotated_frames = []
    
    for frame_idx, frame in enumerate(frames):
        annotated_frame = frame.copy()
        
        # Get all detections for this frame
        for face_result in face_results:
            for detection_info in face_result['detections']:
                if detection_info['frame_idx'] == frame_idx:
                    bbox = detection_info['bbox']
                    score = face_result['mean_score']
                    label = f"Track {face_result['track_id']}"
                    
                    annotated_frame = render_face_overlay(annotated_frame, bbox, score, label)
        
        annotated_frames.append(annotated_frame)
    
    # Save annotated video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annotated_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    
    try:
        save_annotated_video(annotated_frames, annotated_video_path, fps=1)
        print(f"Saved annotated video to {annotated_video_path}")
    except Exception as e:
        warnings.warn(f"Could not save annotated video: {e}")
        annotated_video_path = None
    
    # Convert per_frame_scores from list of lists to flat list of floats
    per_frame_scores_flat = []
    for frame_scores in per_frame_scores:
        if len(frame_scores) > 0:
            per_frame_scores_flat.append(float(np.mean(frame_scores)))
        else:
            per_frame_scores_flat.append(0.5)  # Default neutral score for frames with no faces
    
    return {
        'faces': face_results,
        'per_frame_scores': per_frame_scores_flat,
        'overall_verdict': overall_verdict,
        'overall_score': float(overall_mean),
        'confidence': confidence,
        'consistency': overall_consistency,
        'temporal_stats': {
            'mean': float(overall_mean),
            'std': float(overall_std),
            'median': float(overall_median)
        },
        'annotated_video_path': annotated_video_path
    }
