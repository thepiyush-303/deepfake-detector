#!/usr/bin/env python3
"""
Dataset Preprocessing Script

This script preprocesses datasets for training the deepfake detector.
Supports image, video, and audio modalities with parallel processing.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
import cv2
from tqdm import tqdm
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_utils import extract_frames
from utils.face_utils import FaceDetector
from utils.frequency import compute_fft_spectrum, compute_dct_spectrum
from data.preprocessing import extract_fingerprints, preprocess_image
from data.audio_preprocessing import load_audio, segment_audio, compute_mel_spectrogram, compute_lfcc
from models.srm_kernels import SRMConv2d

warnings.filterwarnings('ignore')


def process_single_image(image_path, output_dir, config):
    """
    Process a single image: detect face, align, extract features.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save processed data
        config: Configuration dictionary
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get relative path for maintaining directory structure
        image_path = Path(image_path)
        
        # Preprocess image (detect and align face)
        aligned_face = preprocess_image(str(image_path), target_size=config['model']['visual']['input_size'])
        
        if aligned_face is None:
            return False, f"No face detected: {image_path.name}"
        
        # Extract forensic fingerprints
        fingerprints = extract_fingerprints(aligned_face)
        
        # Create output path
        output_path = Path(output_dir) / image_path.stem
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save aligned face
        cv2.imwrite(str(output_path.with_suffix('.jpg')), cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
        
        # Save fingerprints
        np.savez_compressed(
            str(output_path.with_suffix('.npz')),
            rgb=fingerprints['rgb'],
            fft=fingerprints['fft'],
            dct=fingerprints['dct'],
            srm=fingerprints['srm']
        )
        
        return True, f"Processed: {image_path.name}"
        
    except Exception as e:
        return False, f"Error processing {image_path.name}: {str(e)}"


def process_single_video(video_path, output_dir, config, face_detector):
    """
    Process a single video: extract frames, detect faces, align.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save processed data
        config: Configuration dictionary
        face_detector: FaceDetector instance
    
    Returns:
        Tuple of (success, message)
    """
    try:
        video_path = Path(video_path)
        
        # Extract frames
        frames = extract_frames(
            str(video_path),
            target_fps=config['video']['target_fps'],
            min_frames=config['video']['min_frames'],
            max_frames=config['video']['max_frames']
        )
        
        if len(frames) == 0:
            return False, f"No frames extracted: {video_path.name}"
        
        # Create output directory for this video
        video_output_dir = Path(output_dir) / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Detect faces
            detection = face_detector.detect_faces(
                frame,
                conf_threshold=config['face_detection']['confidence_threshold'],
                min_size=config['face_detection']['min_face_size'],
                margin=config['face_detection']['margin']
            )
            
            if len(detection['boxes']) == 0:
                continue
            
            # Process first detected face
            bbox = detection['boxes'][0]
            landmarks = detection['landmarks'][0]
            
            # Crop and align face
            x1, y1, x2, y2 = map(int, bbox)
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Resize face
            face_resized = cv2.resize(face_crop, (config['model']['visual']['face_size'], 
                                                   config['model']['visual']['face_size']))
            
            # Extract fingerprints
            fingerprints = extract_fingerprints(face_resized)
            
            # Save face and fingerprints
            frame_output_path = video_output_dir / f"frame_{frame_idx:04d}"
            cv2.imwrite(str(frame_output_path.with_suffix('.jpg')), 
                       cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            
            np.savez_compressed(
                str(frame_output_path.with_suffix('.npz')),
                rgb=fingerprints['rgb'],
                fft=fingerprints['fft'],
                dct=fingerprints['dct'],
                srm=fingerprints['srm'],
                bbox=bbox,
                landmarks=landmarks
            )
            
            processed_count += 1
        
        if processed_count == 0:
            return False, f"No faces detected: {video_path.name}"
        
        return True, f"Processed: {video_path.name} ({processed_count} frames)"
        
    except Exception as e:
        return False, f"Error processing {video_path.name}: {str(e)}"


def process_single_audio(audio_path, output_dir, config):
    """
    Process a single audio file: segment, extract Mel spectrograms and LFCC.
    
    Args:
        audio_path: Path to input audio
        output_dir: Directory to save processed data
        config: Configuration dictionary
    
    Returns:
        Tuple of (success, message)
    """
    try:
        audio_path = Path(audio_path)
        
        # Load audio
        audio_data, sr = load_audio(str(audio_path), target_sr=config['model']['audio']['sample_rate'])
        
        # Segment audio
        segments = segment_audio(
            audio_data,
            sr,
            segment_duration=config['model']['audio']['segment_duration'],
            hop_duration=config['model']['audio']['hop_duration']
        )
        
        if len(segments) == 0:
            return False, f"No segments extracted: {audio_path.name}"
        
        # Create output directory for this audio
        audio_output_dir = Path(output_dir) / audio_path.stem
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each segment
        for seg_idx, segment in enumerate(segments):
            # Compute Mel spectrogram
            mel_spec = compute_mel_spectrogram(
                segment,
                sr,
                n_fft=config['model']['audio']['mel']['n_fft'],
                hop_length=config['model']['audio']['mel']['hop_length'],
                n_mels=config['model']['audio']['mel']['n_mels'],
                fmin=config['model']['audio']['mel']['fmin'],
                fmax=config['model']['audio']['mel']['fmax']
            )
            
            # Compute LFCC
            lfcc = compute_lfcc(
                segment,
                sr,
                n_filters=config['model']['audio']['lfcc']['n_filters'],
                n_lfcc=config['model']['audio']['lfcc']['n_lfcc']
            )
            
            # Save features
            seg_output_path = audio_output_dir / f"segment_{seg_idx:04d}.npz"
            np.savez_compressed(
                str(seg_output_path),
                mel=mel_spec,
                lfcc=lfcc,
                audio=segment
            )
        
        return True, f"Processed: {audio_path.name} ({len(segments)} segments)"
        
    except Exception as e:
        return False, f"Error processing {audio_path.name}: {str(e)}"


def preprocess_images(input_dir, output_dir, config, max_workers=4):
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed data
        config: Configuration dictionary
        max_workers: Number of parallel workers
    """
    print(f"\n{'='*60}")
    print("IMAGE PREPROCESSING")
    print(f"{'='*60}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).rglob(f'*{ext}'))
        image_files.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process images in parallel
    success_count = 0
    fail_count = 0
    
    process_fn = partial(process_single_image, output_dir=output_dir, config=config)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fn, img): img for img in image_files}
        
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
                pbar.set_postfix({'success': success_count, 'failed': fail_count})
    
    print(f"\nCompleted: {success_count} successful, {fail_count} failed")


def preprocess_videos(input_dir, output_dir, config, max_workers=2):
    """
    Preprocess all videos in a directory.
    
    Args:
        input_dir: Input directory containing videos
        output_dir: Output directory for processed data
        config: Configuration dictionary
        max_workers: Number of parallel workers
    """
    print(f"\n{'='*60}")
    print("VIDEO PREPROCESSING")
    print(f"{'='*60}")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(input_dir).rglob(f'*{ext}'))
        video_files.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(video_files)} videos")
    
    if len(video_files) == 0:
        print("No videos found!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize face detector (one per process)
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_detector = FaceDetector(device=device)
    
    # Process videos sequentially (video processing is already intensive)
    success_count = 0
    fail_count = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        success, message = process_single_video(video_file, output_dir, config, face_detector)
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"\n{message}")
    
    print(f"\nCompleted: {success_count} successful, {fail_count} failed")


def preprocess_audio(input_dir, output_dir, config, max_workers=4):
    """
    Preprocess all audio files in a directory.
    
    Args:
        input_dir: Input directory containing audio files
        output_dir: Output directory for processed data
        config: Configuration dictionary
        max_workers: Number of parallel workers
    """
    print(f"\n{'='*60}")
    print("AUDIO PREPROCESSING")
    print(f"{'='*60}")
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).rglob(f'*{ext}'))
        audio_files.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process audio in parallel
    success_count = 0
    fail_count = 0
    
    process_fn = partial(process_single_audio, output_dir=output_dir, config=config)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fn, audio): audio for audio in audio_files}
        
        with tqdm(total=len(audio_files), desc="Processing audio") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
                pbar.set_postfix({'success': success_count, 'failed': fail_count})
    
    print(f"\nCompleted: {success_count} successful, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(description="Preprocess deepfake detection dataset")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing raw data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--modality', type=str, required=True,
                       choices=['image', 'video', 'audio'],
                       help='Data modality to process')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Modality: {args.modality}")
    print(f"Workers: {args.workers}")
    
    # Check input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Preprocess based on modality
    if args.modality == 'image':
        preprocess_images(args.input_dir, args.output_dir, config, args.workers)
    elif args.modality == 'video':
        preprocess_videos(args.input_dir, args.output_dir, config, args.workers)
    elif args.modality == 'audio':
        preprocess_audio(args.input_dir, args.output_dir, config, args.workers)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed data saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
