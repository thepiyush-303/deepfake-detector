# 🔍 Deepfake Detector

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-yellow)

**Advanced GAN Fingerprint-Based Detection for Images, Videos, and Audio**

A state-of-the-art deepfake detection system that analyzes multi-modal content (images, videos, audio) using GAN fingerprint detection and neural network fusion techniques.

---

## 📋 Overview

This project implements a comprehensive deepfake detection pipeline that:

- **Detects deepfake images** by analyzing GAN fingerprints in spectral, noise, and RGB domains
- **Analyzes videos** with face tracking and temporal consistency checking
- **Identifies synthetic audio** using vocoder fingerprint analysis
- **Classifies GAN/vocoder types** to determine the generation method
- **Provides explainability** through GradCAM visualizations

### Key Features

✨ **Multi-Branch Visual Detection**
- RGB branch for semantic features
- Spectral branch for FFT/DCT frequency analysis
- Noise branch for SRM (Steganalysis Rich Model) noise residuals
- Adaptive fusion for combining evidence

🎥 **Temporal Video Analysis**
- Face detection and tracking across frames
- Per-frame and aggregate predictions
- Consistency scoring and confidence estimation

🎵 **Audio Deepfake Detection**
- Mel-spectrogram and LFCC feature extraction
- Segment-level analysis with temporal aggregation
- Vocoder type classification

🌐 **Interactive Web UI**
- Beautiful Gradio interface with dark theme
- Real-time detection with progress tracking
- Visual explanations and detailed analytics

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DEEPFAKE DETECTOR                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   [IMAGE/VIDEO]          [AUDIO]              [FUSION]
        │                     │                     │
    ┌───┴───┐                 │                     │
    │ RGB   │                 │                     │
    │Efficientnet              │                     │
    └───┬───┘                 │                     │
    ┌───┴────┐                │                     │
    │Spectrum│                │                     │
    │FFT+DCT │                │                     │
    └───┬────┘                │                     │
    ┌───┴────┐            ┌───┴────┐           ┌───┴────┐
    │ Noise  │            │  Mel   │           │Adaptive│
    │  SRM   │            │ +LFCC  │           │Fusion  │
    └───┬────┘            └───┬────┘           └───┬────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Binary + GAN/    │
                    │  Vocoder Classifier│
                    └───────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/thepiyush-303/deepfake-detector.git
cd deepfake-detector
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Optional: GPU Setup**

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ⚡ Quick Start

### Detect Deepfakes in an Image

```python
from inference.predict_image import predict_image, load_model

# Load model (with or without checkpoint)
model = load_model(checkpoint_path='checkpoints/visual_model.pth')

# Run prediction
result = predict_image('path/to/image.jpg', model, device='cuda')

print(f"Verdict: {result['verdict']}")
print(f"Fake Probability: {result['fakeness_score']:.2%}")
print(f"Most Likely GAN: {result['gan_type']}")
```

---

## 💻 Usage

### 1. Web UI (Recommended for Interactive Use)

Launch the Gradio web interface:

```bash
python -m ui.app
```

Then open your browser to `http://localhost:7860`

The UI provides three tabs:
- **📸 Image Detection**: Upload images for instant analysis
- **🎥 Video Detection**: Analyze videos with temporal tracking
- **🎵 Audio Detection**: Detect synthetic voices and vocoders

### 2. Command-Line Image Prediction

```python
from inference.predict_image import predict_image, load_model
import json

model = load_model(checkpoint_path='checkpoints/best_visual.pth', device='cuda')
result = predict_image('test_image.jpg', model)

print(json.dumps(result, indent=2, default=str))
```

### 3. Command-Line Video Prediction

```python
from inference.predict_video import predict_video
from inference.predict_image import load_model

model = load_model(checkpoint_path='checkpoints/best_visual.pth', device='cuda')
result = predict_video('test_video.mp4', model, output_dir='output')

print(f"Overall Verdict: {result['overall_verdict']}")
print(f"Confidence: {result['confidence']}")
print(f"Consistency: {result['consistency']}")
print(f"Annotated video saved to: {result['annotated_video_path']}")
```

### 4. Command-Line Audio Prediction

```python
from inference.predict_audio import predict_audio, load_audio_model

model = load_audio_model(checkpoint_path='checkpoints/best_audio.pth', device='cuda')
result = predict_audio('test_audio.wav', model)

print(f"Verdict: {result['verdict']}")
print(f"Vocoder Type: {result['vocoder_type']}")
print(f"Temporal Consistency: {result['consistency']}")
```

### 5. Training from Scratch

**Train Visual Model:**
```bash
python -m training.train_visual \
    --data_dir data/train \
    --val_dir data/val \
    --config config/default.yaml \
    --output_dir checkpoints/visual \
    --epochs 20
```

**Train Audio Model:**
```bash
python -m training.train_audio \
    --data_dir data/audio_train \
    --val_dir data/audio_val \
    --config config/default.yaml \
    --output_dir checkpoints/audio \
    --epochs 15
```

### 6. Dataset Preprocessing

**Preprocess Images:**
```bash
python scripts/preprocess_dataset.py \
    --input_dir raw_data/images \
    --output_dir processed_data/images \
    --modality image \
    --config config/default.yaml \
    --workers 4
```

**Preprocess Videos:**
```bash
python scripts/preprocess_dataset.py \
    --input_dir raw_data/videos \
    --output_dir processed_data/videos \
    --modality video \
    --config config/default.yaml \
    --workers 2
```

**Preprocess Audio:**
```bash
python scripts/preprocess_dataset.py \
    --input_dir raw_data/audio \
    --output_dir processed_data/audio \
    --modality audio \
    --config config/default.yaml \
    --workers 4
```

### 7. Model Evaluation

**Evaluate Visual Model:**
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_visual.pth \
    --data_dir test_data/images \
    --modality visual \
    --config config/default.yaml \
    --output_dir evaluation_results/visual
```

**Evaluate Audio Model:**
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_audio.pth \
    --data_dir test_data/audio \
    --modality audio \
    --config config/default.yaml \
    --output_dir evaluation_results/audio
```

---

## 📁 Project Structure

```
deepfake-detector/
├── config/
│   └── default.yaml          # Configuration file
├── data/
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset classes
│   ├── preprocessing.py      # Image preprocessing utilities
│   ├── audio_preprocessing.py # Audio preprocessing utilities
│   └── augmentation.py       # Data augmentation
├── models/
│   ├── __init__.py
│   ├── fusion_model.py       # Visual multi-branch fusion model
│   ├── audio_model.py        # Audio deepfake detection model
│   ├── rgb_branch.py         # RGB feature extractor
│   ├── spectrum_branch.py    # Spectral feature extractor
│   ├── noise_branch.py       # Noise residual extractor
│   └── srm_kernels.py        # SRM filter kernels
├── training/
│   ├── __init__.py
│   ├── train_visual.py       # Visual model training script
│   ├── train_audio.py        # Audio model training script
│   ├── losses.py             # Loss functions
│   └── scheduler.py          # Learning rate schedulers
├── inference/
│   ├── __init__.py
│   ├── predict_image.py      # Image prediction
│   ├── predict_video.py      # Video prediction with tracking
│   ├── predict_audio.py      # Audio prediction
│   └── explainability.py     # GradCAM visualizations
├── utils/
│   ├── __init__.py
│   ├── face_utils.py         # Face detection (MTCNN)
│   ├── video_utils.py        # Video processing utilities
│   ├── frequency.py          # FFT/DCT spectrum computation
│   └── metrics.py            # Evaluation metrics (AUC, EER, AP)
├── ui/
│   ├── __init__.py
│   └── app.py                # Gradio web interface
├── scripts/
│   ├── __init__.py
│   ├── preprocess_dataset.py # Dataset preprocessing CLI
│   └── evaluate.py           # Model evaluation CLI
├── tests/
│   ├── test_data_modules.py  # Unit tests for data modules
│   └── test_pipeline.py      # End-to-end integration tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🧠 Model Details

### Visual Model Architecture

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 256×256 pixels
- **Branches**:
  - **RGB Branch**: Processes raw RGB image
  - **Spectral Branch**: FFT + DCT frequency analysis (2 channels)
  - **Noise Branch**: SRM noise residuals (30 filters)
- **Fusion**: Adaptive attention-based fusion
- **Output**:
  - Binary classification (Real/Fake)
  - 7-class GAN type classification

### Audio Model Architecture

- **Backbone**: ResNet-18 (adapted for audio)
- **Sample Rate**: 16kHz
- **Features**:
  - Mel-spectrogram (80 mel bins)
  - LFCC (Linear Frequency Cepstral Coefficients, 40 coefficients)
- **Segmentation**: 4-second segments with 2-second hop
- **Output**:
  - Binary classification (Real/Fake)
  - 7-class vocoder type classification

---

## 🎯 Supported GAN Types

| Class ID | GAN Type    | Description                      |
|----------|-------------|----------------------------------|
| 0        | Real        | Authentic/non-synthesized images |
| 1        | ProGAN      | Progressive GAN                  |
| 2        | StyleGAN    | StyleGAN v1/v2                   |
| 3        | StarGAN     | StarGAN (attribute transfer)     |
| 4        | CycleGAN    | CycleGAN (unpaired translation)  |
| 5        | Deepfakes   | Traditional deepfake methods     |
| 6        | Unknown     | Unidentified GAN architecture    |

---

## 🎙️ Supported Vocoder Types

| Class ID | Vocoder Type       | Description                |
|----------|-------------------|----------------------------|
| 0        | Real              | Natural human voice        |
| 1        | WaveGlow          | Flow-based vocoder         |
| 2        | WaveRNN           | Recurrent neural vocoder   |
| 3        | MelGAN            | GAN-based vocoder          |
| 4        | HiFi-GAN          | High-fidelity GAN vocoder  |
| 5        | TTS               | Generic TTS systems        |
| 6        | Unknown           | Unidentified vocoder       |

---

## ⚙️ Configuration

Key parameters in `config/default.yaml`:

### Model Configuration
```yaml
model:
  visual:
    backbone: "efficientnet_b0"
    input_size: 256
    face_size: 224
    num_gan_types: 7
  
  audio:
    sample_rate: 16000
    segment_duration: 4.0
    hop_duration: 2.0
```

### Training Configuration
```yaml
training:
  visual:
    batch_size: 16
    gradient_accumulation: 4
    lr_adapter: 1.0e-3
    lr_head: 1.0e-3
    
  audio:
    batch_size: 32
    gradient_accumulation: 2
    lr_backbone: 1.0e-4
```

### Inference Configuration
```yaml
inference:
  device: "cuda"  # or "cpu"
  batch_size: 16
  threshold: 0.5
```

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_pipeline.py -v      # Integration tests
pytest tests/test_data_modules.py -v  # Unit tests
```

Run tests with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run linters
black .
flake8 .

# Run tests
pytest tests/ -v
```

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Deepfake Detector Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{deepfake_detector,
  title={Deepfake Detector: GAN Fingerprint-Based Multi-Modal Detection},
  author={Deepfake Detector Contributors},
  year={2024},
  url={https://github.com/thepiyush-303/deepfake-detector}
}
```

---

## 🙏 Acknowledgments

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- SRM Filters: [Fridrich & Kodovsky, 2012](https://ieeexplore.ieee.org/document/6197267)
- MTCNN: [Zhang et al., 2016](https://arxiv.org/abs/1604.02878)
- Gradio: [Abid et al., 2019](https://gradio.app/)

---

## 📧 Contact

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Create an issue](https://github.com/thepiyush-303/deepfake-detector/issues)
- **Discussions**: [Join the discussion](https://github.com/thepiyush-303/deepfake-detector/discussions)

---

<div align="center">
  <strong>⭐ Star this repository if you find it useful! ⭐</strong>
  
  Made with ❤️ by the Deepfake Detector team
</div>
