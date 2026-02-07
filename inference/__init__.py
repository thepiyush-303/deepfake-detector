"""Inference module initialization."""

from .predict_image import load_model, predict_image, visualize_results
from .predict_video import predict_video
from .predict_audio import load_audio_model, predict_audio
from .explainability import GradCAM, generate_gradcam, visualize_all_branches

__all__ = [
    'load_model',
    'predict_image',
    'visualize_results',
    'predict_video',
    'load_audio_model',
    'predict_audio',
    'GradCAM',
    'generate_gradcam',
    'visualize_all_branches'
]
