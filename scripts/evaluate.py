#!/usr/bin/env python3
"""
Model Evaluation Script

This script evaluates trained deepfake detection models on test datasets.
Computes metrics, generates visualizations, and supports cross-GAN evaluation.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_score, recall_score
)
from tqdm import tqdm
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import DeepfakeFusionModel
from models.audio_model import AudioDeepfakeModel
from inference.predict_image import predict_image, load_model
from inference.predict_audio import predict_audio, load_audio_model
from utils.metrics import compute_auc, compute_eer, compute_ap

warnings.filterwarnings('ignore')


def evaluate_visual_model(model, data_dir, config, device='cuda'):
    """
    Evaluate visual model on test dataset.
    
    Args:
        model: DeepfakeFusionModel instance
        data_dir: Directory containing test images
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        Dictionary of evaluation results
    """
    print("Evaluating visual model...")
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(data_dir).rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found!")
        return None
    
    # Collect predictions and ground truth
    y_true = []
    y_scores = []
    y_pred = []
    gan_type_true = []
    gan_type_pred = []
    failed_count = 0
    
    # Determine labels from directory structure
    # Expected structure: data_dir/real/*.jpg or data_dir/fake/*.jpg
    # Or: data_dir/real/gan_type/*.jpg
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Determine ground truth from path
            parts = image_file.parts
            if 'real' in str(image_file).lower():
                true_label = 0  # Real
                true_gan_type = 'Real'
            elif 'fake' in str(image_file).lower():
                true_label = 1  # Fake
                # Try to determine GAN type from path
                gan_types = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'BigGAN', 'CycleGAN', 'StarGAN', 'GauGAN']
                true_gan_type = 'Unknown'
                for gan in gan_types:
                    if gan.lower() in str(image_file).lower():
                        true_gan_type = gan
                        break
            else:
                # Skip if we can't determine label
                continue
            
            # Run prediction
            result = predict_image(str(image_file), model, device=device)
            
            y_true.append(true_label)
            y_scores.append(result['fakeness_score'])
            y_pred.append(1 if result['verdict'] == 'FAKE' else 0)
            gan_type_true.append(true_gan_type)
            gan_type_pred.append(result['gan_type'])
            
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Processed {len(y_true)} images, {failed_count} failed")
    
    if len(y_true) == 0:
        print("No valid predictions!")
        return None
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    try:
        auc = compute_auc(y_true, y_scores)
    except:
        auc = 0.0
    
    try:
        eer = compute_eer(y_true, y_scores)
    except:
        eer = 0.0
    
    try:
        ap = compute_ap(y_true, y_scores)
    except:
        ap = 0.0
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Per-GAN-type accuracy
    gan_type_accuracy = {}
    unique_gan_types = set(gan_type_true)
    for gan_type in unique_gan_types:
        if gan_type == 'Real':
            continue
        indices = [i for i, gt in enumerate(gan_type_true) if gt == gan_type]
        if len(indices) > 0:
            gan_preds = [y_pred[i] for i in indices]
            gan_true = [y_true[i] for i in indices]
            gan_type_accuracy[gan_type] = accuracy_score(gan_true, gan_preds)
    
    return {
        'auc': auc,
        'eer': eer,
        'ap': ap,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_scores': y_scores,
        'y_pred': y_pred,
        'gan_type_accuracy': gan_type_accuracy,
        'num_samples': len(y_true)
    }


def evaluate_audio_model(model, data_dir, config, device='cuda'):
    """
    Evaluate audio model on test dataset.
    
    Args:
        model: AudioDeepfakeModel instance
        data_dir: Directory containing test audio files
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        Dictionary of evaluation results
    """
    print("Evaluating audio model...")
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(data_dir).rglob(f'*{ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return None
    
    # Collect predictions and ground truth
    y_true = []
    y_scores = []
    y_pred = []
    failed_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        try:
            # Determine ground truth from path
            if 'real' in str(audio_file).lower():
                true_label = 0  # Real
            elif 'fake' in str(audio_file).lower():
                true_label = 1  # Fake
            else:
                continue
            
            # Run prediction
            result = predict_audio(str(audio_file), model, device=device)
            
            y_true.append(true_label)
            y_scores.append(result['fakeness_score'])
            y_pred.append(1 if result['verdict'] == 'FAKE' else 0)
            
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Processed {len(y_true)} audio files, {failed_count} failed")
    
    if len(y_true) == 0:
        print("No valid predictions!")
        return None
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    try:
        auc = compute_auc(y_true, y_scores)
    except:
        auc = 0.0
    
    try:
        eer = compute_eer(y_true, y_scores)
    except:
        eer = 0.0
    
    try:
        ap = compute_ap(y_true, y_scores)
    except:
        ap = 0.0
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'auc': auc,
        'eer': eer,
        'ap': ap,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_scores': y_scores,
        'y_pred': y_pred,
        'num_samples': len(y_true)
    }


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(y_true, y_scores, auc, output_path):
    """Generate and save ROC curve plot."""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2563eb', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2, color='#2563eb')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_path}")


def save_classification_report(y_true, y_pred, output_path):
    """Generate and save classification report."""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Real', 'Fake'],
                                   digits=4)
    
    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"Saved classification report to {output_path}")


def print_summary(results, modality):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print(f"{modality.upper()} MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"AUC:               {results['auc']:.4f}")
    print(f"EER:               {results['eer']:.4f}")
    print(f"AP:                {results['ap']:.4f}")
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"F1 Score:          {results['f1']:.4f}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    
    if 'gan_type_accuracy' in results and results['gan_type_accuracy']:
        print("\nPer-GAN-Type Accuracy:")
        print("-" * 60)
        for gan_type, acc in sorted(results['gan_type_accuracy'].items()):
            print(f"  {gan_type:15s}: {acc:.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument('--checkpoint', type=str, required=False,
                       help='Path to model checkpoint (optional, will use random weights if not provided)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--modality', type=str, required=True,
                       choices=['visual', 'audio'],
                       help='Model modality to evaluate')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluation Configuration:")
    print(f"  Checkpoint: {args.checkpoint if args.checkpoint else 'None (random weights)'}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Modality: {args.modality}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.output_dir}")
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # Load model and evaluate
    if args.modality == 'visual':
        model = load_model(checkpoint_path=args.checkpoint, device=args.device)
        results = evaluate_visual_model(model, args.data_dir, config, device=args.device)
    elif args.modality == 'audio':
        model = load_audio_model(checkpoint_path=args.checkpoint, device=args.device)
        results = evaluate_audio_model(model, args.data_dir, config, device=args.device)
    
    if results is None:
        print("Evaluation failed!")
        return
    
    # Print summary
    print_summary(results, args.modality)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        results['y_true'], 
        results['y_pred'],
        output_dir / 'confusion_matrix.png'
    )
    
    # ROC curve
    plot_roc_curve(
        results['y_true'],
        results['y_scores'],
        results['auc'],
        output_dir / 'roc_curve.png'
    )
    
    # Classification report
    save_classification_report(
        results['y_true'],
        results['y_pred'],
        output_dir / 'classification_report.txt'
    )
    
    # Save metrics to file
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"{args.modality.upper()} Model Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of samples: {results['num_samples']}\n")
        f.write(f"AUC:               {results['auc']:.4f}\n")
        f.write(f"EER:               {results['eer']:.4f}\n")
        f.write(f"AP:                {results['ap']:.4f}\n")
        f.write(f"Accuracy:          {results['accuracy']:.4f}\n")
        f.write(f"F1 Score:          {results['f1']:.4f}\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        
        if 'gan_type_accuracy' in results and results['gan_type_accuracy']:
            f.write("\nPer-GAN-Type Accuracy:\n")
            f.write("-" * 60 + "\n")
            for gan_type, acc in sorted(results['gan_type_accuracy'].items()):
                f.write(f"  {gan_type:15s}: {acc:.4f}\n")
    
    print(f"Saved metrics to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
