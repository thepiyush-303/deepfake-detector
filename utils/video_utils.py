"""
Video processing utilities for frame extraction and face tracking.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def extract_frames(video_path, target_fps=1, min_frames=8, max_frames=32):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        target_fps: Target frames per second for extraction (default: 1)
        min_frames: Minimum number of frames to extract (default: 8)
        max_frames: Maximum number of frames to extract (default: 32)
    
    Returns:
        List of frames as numpy arrays (RGB format)
    """
    if not video_path:
        raise ValueError("Video path cannot be empty")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("Invalid video properties")
    
    # Calculate frame sampling interval
    frame_interval = max(1, int(fps / target_fps))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Sample frames at target_fps
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Stop if we've reached max_frames
            if len(frames) >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    
    # Ensure we have at least min_frames
    if len(frames) < min_frames:
        # If we don't have enough frames, try to extract more uniformly
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Extract frames uniformly across the video
        indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
    
    return frames


def track_faces(frames, face_bboxes):
    """
    Track faces across frames using IoU-based tracking.
    
    Args:
        frames: List of video frames
        face_bboxes: List of face bounding boxes per frame, each element is a list of [x1, y1, x2, y2]
    
    Returns:
        List of track IDs for each detection, shape matches face_bboxes structure
    """
    iou_threshold = 0.3
    
    if not face_bboxes or len(face_bboxes) == 0:
        return []
    
    # Initialize tracking
    tracks = []
    track_results = []
    next_track_id = 0
    
    for frame_idx, bboxes in enumerate(face_bboxes):
        if frame_idx == 0:
            # Initialize tracks for first frame
            frame_tracks = []
            for bbox in bboxes:
                tracks.append({
                    'id': next_track_id,
                    'bbox': bbox,
                    'last_frame': 0
                })
                frame_tracks.append(next_track_id)
                next_track_id += 1
        else:
            # Match current detections with existing tracks
            frame_tracks = []
            matched_tracks = set()
            
            for bbox in bboxes:
                best_iou = 0
                best_track_id = -1
                
                # Find best matching track
                for track in tracks:
                    if track['id'] in matched_tracks:
                        continue
                    
                    iou = compute_iou(bbox, track['bbox'])
                    
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_track_id = track['id']
                
                if best_track_id >= 0:
                    # Match found
                    frame_tracks.append(best_track_id)
                    matched_tracks.add(best_track_id)
                    
                    # Update track
                    for track in tracks:
                        if track['id'] == best_track_id:
                            track['bbox'] = bbox
                            track['last_frame'] = frame_idx
                            break
                else:
                    # New track
                    tracks.append({
                        'id': next_track_id,
                        'bbox': bbox,
                        'last_frame': frame_idx
                    })
                    frame_tracks.append(next_track_id)
                    next_track_id += 1
        
        track_results.append(frame_tracks)
    
    return track_results


def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def render_face_overlay(frame, bbox, score, label):
    """
    Draw colored bounding box and label on frame.
    
    Args:
        frame: Video frame as numpy array (RGB or BGR)
        bbox: Bounding box [x1, y1, x2, y2]
        score: Confidence score (0-1), used for color gradient
        label: Text label to display
    
    Returns:
        Frame with overlay drawn
    """
    frame_copy = frame.copy()
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Color gradient from green (0.0) to red (1.0)
    # Green: (0, 255, 0), Red: (255, 0, 0)
    red = int(255 * score)
    green = int(255 * (1 - score))
    color = (red, green, 0)
    
    # Draw bounding box
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    label_text = f"{label}: {score:.2f}"
    
    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame_copy,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame_copy,
        label_text,
        (x1, y1 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )
    
    return frame_copy


def save_annotated_video(frames, output_path, fps=30):
    """
    Save annotated frames as a video file.
    
    Args:
        frames: List of annotated frames (RGB format)
        output_path: Path to save the output video
        fps: Frames per second for output video (default: 30)
    """
    if not frames or len(frames) == 0:
        raise ValueError("No frames to save")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
