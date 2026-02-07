"""
Gradio Web Application for Deepfake Detection

This module provides a web interface for detecting deepfakes in images, videos, and audio files.
"""

import os
import sys
import yaml
import warnings
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict_image import predict_image, load_model
from inference.predict_video import predict_video
from inference.predict_audio import predict_audio, load_audio_model
from inference.explainability import generate_gradcam

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
DEFAULT_NO_FACE_SCORE = 0.5  # Default score for frames with no detected faces

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default.yaml')

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Initialize models (without checkpoints for now)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
visual_model = None
audio_model = None


def initialize_models():
    """Initialize models on first use."""
    global visual_model, audio_model
    
    if visual_model is None:
        try:
            # Try to load checkpoint from config
            visual_checkpoint_path = None
            checkpoints_dir = config.get('paths', {}).get('checkpoints', '')
            if checkpoints_dir:
                visual_checkpoint_file = os.path.join(checkpoints_dir, 'best_visual.pth')
                if os.path.exists(visual_checkpoint_file):
                    visual_checkpoint_path = visual_checkpoint_file
            
            visual_model = load_model(checkpoint_path=visual_checkpoint_path, device=device)
            print("Visual model initialized")
        except Exception as e:
            print(f"Error initializing visual model: {e}")
    
    if audio_model is None:
        try:
            # Try to load checkpoint from config
            audio_checkpoint_path = None
            checkpoints_dir = config.get('paths', {}).get('checkpoints', '')
            if checkpoints_dir:
                audio_checkpoint_file = os.path.join(checkpoints_dir, 'best_audio.pth')
                if os.path.exists(audio_checkpoint_file):
                    audio_checkpoint_path = audio_checkpoint_file
            
            audio_model = load_audio_model(checkpoint_path=audio_checkpoint_path, device=device)
            print("Audio model initialized")
        except Exception as e:
            print(f"Error initializing audio model: {e}")


def detect_image(image_file):
    """
    Process image and return detection results.
    
    Args:
        image_file: Uploaded image file
    
    Returns:
        Tuple of (image, verdict_html, gan_type_html, heatmap)
    """
    if image_file is None:
        return None, "⚠️ Please upload an image", "", None
    
    try:
        initialize_models()
        
        if visual_model is None:
            return None, "❌ Model not available", "", None
        
        # Save temporary file
        temp_path = "/tmp/temp_image.jpg"
        Image.open(image_file).save(temp_path)
        
        # Run prediction
        result = predict_image(temp_path, visual_model, device=device)
        
        # Create verdict display
        verdict = result['verdict']
        fakeness = result['fakeness_score']
        confidence = result['confidence']
        face_detected = result.get('face_detected', True)
        model_trained = result.get('model_trained', False)
        
        # Force LOW confidence if model is untrained
        if not model_trained:
            confidence = 'LOW'
        
        if verdict == 'FAKE':
            verdict_color = "#ff4444"
            verdict_emoji = "🚨"
        else:
            verdict_color = "#44ff44"
            verdict_emoji = "✅"
        
        # Add warning if model is untrained
        untrained_warning = ""
        if not model_trained:
            untrained_warning = '''
            <div style="padding: 10px; background: #332200; border: 1px solid #ff9900; border-radius: 8px; margin: 10px 0;">
                <p style="color: #ff9900; font-size: 14px; margin: 0;">
                    ⚠️ <strong>Untrained Model</strong>: No checkpoint loaded. Predictions are random and unreliable. 
                    Train the model or provide a checkpoint path for accurate results.
                </p>
            </div>
            '''
        
        # Add warning if no face was detected
        warning_text = ""
        if not face_detected:
            warning_text = '<p style="color: #ff9900; font-size: 14px; margin: 10px 0;">⚠️ No face detected — analyzing full image</p>'
        
        verdict_html = f"""
        <div style="text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px; margin: 10px 0;">
            {untrained_warning}
            <h2 style="color: {verdict_color}; margin: 0;">{verdict_emoji} {verdict}</h2>
            {warning_text}
            <p style="color: #cccccc; font-size: 18px; margin: 10px 0;">
                Fake Probability: <span style="color: {verdict_color}; font-weight: bold;">{fakeness:.1%}</span>
            </p>
            <p style="color: #aaaaaa; font-size: 14px; margin: 5px 0;">
                Confidence: {confidence}
            </p>
        </div>
        """
        
        # Create GAN type display
        gan_type = result['gan_type']
        gan_probs = result['gan_probs']
        
        gan_html = f"""
        <div style="padding: 15px; background: #1a1a1a; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #ffffff; margin-top: 0;">Most Likely GAN: {gan_type}</h3>
            <div style="color: #cccccc;">
        """
        
        # Sort GAN types by probability
        sorted_gans = sorted(gan_probs.items(), key=lambda x: x[1], reverse=True)
        for gan, prob in sorted_gans[:5]:
            bar_width = int(prob * 100)
            gan_html += f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span>{gan}</span>
                    <span>{prob:.1%}</span>
                </div>
                <div style="background: #333; border-radius: 5px; height: 20px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #4a9eff, #7b68ee); 
                         width: {bar_width}%; height: 100%; transition: width 0.3s;"></div>
                </div>
            </div>
            """
        
        gan_html += "</div></div>"
        
        # Generate GradCAM heatmap
        try:
            gradcam_result = generate_gradcam(
                visual_model, 
                result['image'], 
                target_class=None, 
                branch='spectrum'
            )
            heatmap = gradcam_result['overlay']
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            heatmap_pil = None
        
        # Return aligned face image
        face_image = Image.fromarray(result['image'].astype(np.uint8))
        
        return face_image, verdict_html, gan_html, heatmap_pil
        
    except Exception as e:
        return None, f"❌ Error processing image: {str(e)}", "", None


def detect_video(video_file, progress=gr.Progress()):
    """
    Process video and return detection results.
    
    Args:
        video_file: Uploaded video file
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (verdict_html, timeline_plot, preview_image, gan_type_html)
    """
    if video_file is None:
        return "⚠️ Please upload a video", None, None, ""
    
    try:
        initialize_models()
        
        if visual_model is None:
            return "❌ Model not available", None, None, ""
        
        progress(0, desc="Starting video analysis...")
        
        # Run prediction
        result = predict_video(video_file, visual_model, device=device, output_dir='/tmp')
        
        progress(0.5, desc="Generating visualizations...")
        
        # Create verdict display
        verdict = result['overall_verdict']
        fakeness = result['overall_score']
        confidence = result['confidence']
        consistency = result['consistency']
        model_trained = result.get('model_trained', False)
        
        # Force LOW confidence if model is untrained
        if not model_trained:
            confidence = 'LOW'
        
        if verdict == 'FAKE':
            verdict_color = "#ff4444"
            verdict_emoji = "🚨"
        else:
            verdict_color = "#44ff44"
            verdict_emoji = "✅"
        
        # Add warning if model is untrained
        untrained_warning = ""
        if not model_trained:
            untrained_warning = '''
            <div style="padding: 10px; background: #332200; border: 1px solid #ff9900; border-radius: 8px; margin: 10px 0;">
                <p style="color: #ff9900; font-size: 14px; margin: 0;">
                    ⚠️ <strong>Untrained Model</strong>: No checkpoint loaded. Predictions are random and unreliable. 
                    Train the model or provide a checkpoint path for accurate results.
                </p>
            </div>
            '''
        
        verdict_html = f"""
        <div style="text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px; margin: 10px 0;">
            {untrained_warning}
            <h2 style="color: {verdict_color}; margin: 0;">{verdict_emoji} {verdict}</h2>
            <p style="color: #cccccc; font-size: 18px; margin: 10px 0;">
                Overall Fake Probability: <span style="color: {verdict_color}; font-weight: bold;">{fakeness:.1%}</span>
            </p>
            <p style="color: #aaaaaa; font-size: 14px; margin: 5px 0;">
                Confidence: {confidence} | Consistency: {consistency}
            </p>
            <p style="color: #999999; font-size: 12px; margin: 5px 0;">
                Analyzed {len(result['faces'])} face(s) across {len(result['per_frame_scores'])} frames
            </p>
        </div>
        """
        
        # Create timeline chart
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1a1a1a')
        ax.set_facecolor('#0d0d0d')
        
        frames = list(range(len(result['per_frame_scores'])))
        scores = result['per_frame_scores']
        
        # Ensure scores is a flat list of numbers (safety check for jagged arrays)
        if scores and isinstance(scores[0], (list, np.ndarray)):
            scores = [float(np.mean(s)) if len(s) > 0 else DEFAULT_NO_FACE_SCORE for s in scores]
        
        ax.plot(frames, scores, color='#4a9eff', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0.5, color='#ff4444', linestyle='--', linewidth=1, alpha=0.7, label='Threshold')
        ax.fill_between(frames, scores, 0.5, where=[s > 0.5 for s in scores], 
                        color='#ff4444', alpha=0.3, label='Fake')
        ax.fill_between(frames, scores, 0.5, where=[s <= 0.5 for s in scores], 
                        color='#44ff44', alpha=0.3, label='Real')
        
        ax.set_xlabel('Frame Number', color='#cccccc', fontsize=12)
        ax.set_ylabel('Fake Probability', color='#cccccc', fontsize=12)
        ax.set_title('Per-Frame Detection Timeline', color='#ffffff', fontsize=14, pad=15)
        ax.tick_params(colors='#cccccc')
        ax.spines['bottom'].set_color('#444444')
        ax.spines['top'].set_color('#444444')
        ax.spines['left'].set_color('#444444')
        ax.spines['right'].set_color('#444444')
        ax.legend(facecolor='#1a1a1a', edgecolor='#444444', labelcolor='#cccccc')
        ax.grid(True, alpha=0.2, color='#444444')
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # Create GAN type display for primary face
        if result['faces']:
            primary_face = result['faces'][0]
            gan_type = primary_face['gan_type']
            gan_probs = primary_face.get('gan_probs', {})
            
            gan_html = f"""
            <div style="padding: 15px; background: #1a1a1a; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: #ffffff; margin-top: 0;">Primary Face - Most Likely GAN: {gan_type}</h3>
                <div style="color: #cccccc;">
            """
            
            sorted_gans = sorted(gan_probs.items(), key=lambda x: x[1], reverse=True)
            for gan, prob in sorted_gans[:5]:
                bar_width = int(prob * 100)
                gan_html += f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                        <span>{gan}</span>
                        <span>{prob:.1%}</span>
                    </div>
                    <div style="background: #333; border-radius: 5px; height: 20px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #4a9eff, #7b68ee); 
                             width: {bar_width}%; height: 100%; transition: width 0.3s;"></div>
                    </div>
                </div>
                """
            
            gan_html += "</div></div>"
        else:
            gan_html = "<p style='color: #cccccc;'>No faces detected</p>"
        
        # Load preview from annotated video (first frame if available)
        preview_image = None
        if os.path.exists(result['annotated_video_path']):
            import cv2
            cap = cv2.VideoCapture(result['annotated_video_path'])
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_image = Image.fromarray(frame_rgb)
            cap.release()
        
        progress(1.0, desc="Complete!")
        
        return verdict_html, fig, preview_image, gan_html
        
    except Exception as e:
        return f"❌ Error processing video: {str(e)}", None, None, ""


def detect_audio(audio_file, progress=gr.Progress()):
    """
    Process audio and return detection results.
    
    Args:
        audio_file: Uploaded audio file
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (verdict_html, segment_plot, spectrogram_plot, vocoder_html)
    """
    if audio_file is None:
        return "⚠️ Please upload an audio file", None, None, ""
    
    try:
        initialize_models()
        
        if audio_model is None:
            return "❌ Model not available", None, None, ""
        
        progress(0, desc="Starting audio analysis...")
        
        # Run prediction
        result = predict_audio(audio_file, audio_model, device=device)
        
        progress(0.5, desc="Generating visualizations...")
        
        # Create verdict display
        verdict = result['verdict']
        fakeness = result['fakeness_score']
        confidence = result['confidence']
        consistency = result['consistency']
        model_trained = result.get('model_trained', False)
        
        # Force LOW confidence if model is untrained
        if not model_trained:
            confidence = 'LOW'
        
        if verdict == 'FAKE':
            verdict_color = "#ff4444"
            verdict_emoji = "🚨"
        else:
            verdict_color = "#44ff44"
            verdict_emoji = "✅"
        
        # Add warning if model is untrained
        untrained_warning = ""
        if not model_trained:
            untrained_warning = '''
            <div style="padding: 10px; background: #332200; border: 1px solid #ff9900; border-radius: 8px; margin: 10px 0;">
                <p style="color: #ff9900; font-size: 14px; margin: 0;">
                    ⚠️ <strong>Untrained Model</strong>: No checkpoint loaded. Predictions are random and unreliable. 
                    Train the model or provide a checkpoint path for accurate results.
                </p>
            </div>
            '''
        
        verdict_html = f"""
        <div style="text-align: center; padding: 20px; background: #1a1a1a; border-radius: 10px; margin: 10px 0;">
            {untrained_warning}
            <h2 style="color: {verdict_color}; margin: 0;">{verdict_emoji} {verdict}</h2>
            <p style="color: #cccccc; font-size: 18px; margin: 10px 0;">
                Fake Probability: <span style="color: {verdict_color}; font-weight: bold;">{fakeness:.1%}</span>
            </p>
            <p style="color: #aaaaaa; font-size: 14px; margin: 5px 0;">
                Confidence: {confidence} | Consistency: {consistency}
            </p>
            <p style="color: #999999; font-size: 12px; margin: 5px 0;">
                Analyzed {len(result['per_segment_scores'])} segment(s)
            </p>
        </div>
        """
        
        # Create per-segment scores plot
        fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor='#1a1a1a')
        ax1.set_facecolor('#0d0d0d')
        
        segments = list(range(len(result['per_segment_scores'])))
        scores = result['per_segment_scores']
        
        colors = ['#ff4444' if s > 0.5 else '#44ff44' for s in scores]
        ax1.bar(segments, scores, color=colors, alpha=0.7, edgecolor='#ffffff', linewidth=0.5)
        ax1.axhline(y=0.5, color='#ffaa00', linestyle='--', linewidth=2, alpha=0.8, label='Threshold')
        
        ax1.set_xlabel('Segment Number', color='#cccccc', fontsize=12)
        ax1.set_ylabel('Fake Probability', color='#cccccc', fontsize=12)
        ax1.set_title('Per-Segment Detection Scores', color='#ffffff', fontsize=14, pad=15)
        ax1.tick_params(colors='#cccccc')
        ax1.spines['bottom'].set_color('#444444')
        ax1.spines['top'].set_color('#444444')
        ax1.spines['left'].set_color('#444444')
        ax1.spines['right'].set_color('#444444')
        ax1.legend(facecolor='#1a1a1a', edgecolor='#444444', labelcolor='#cccccc')
        ax1.grid(True, alpha=0.2, color='#444444', axis='y')
        ax1.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # Create spectrogram visualization
        try:
            from data.audio_preprocessing import load_audio, compute_mel_spectrogram
            audio_data, sr = load_audio(audio_file, target_sr=16000)
            mel_spec = compute_mel_spectrogram(audio_data, sr)
            
            fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor='#1a1a1a')
            ax2.set_facecolor('#0d0d0d')
            
            im = ax2.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_xlabel('Time', color='#cccccc', fontsize=12)
            ax2.set_ylabel('Mel Frequency', color='#cccccc', fontsize=12)
            ax2.set_title('Mel Spectrogram', color='#ffffff', fontsize=14, pad=15)
            ax2.tick_params(colors='#cccccc')
            
            cbar = plt.colorbar(im, ax=ax2)
            cbar.ax.tick_params(colors='#cccccc')
            cbar.set_label('Amplitude (dB)', color='#cccccc')
            
            plt.tight_layout()
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
            fig2 = None
        
        # Create vocoder type display
        vocoder_type = result['vocoder_type']
        vocoder_probs = result['vocoder_probs']
        
        vocoder_html = f"""
        <div style="padding: 15px; background: #1a1a1a; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #ffffff; margin-top: 0;">Most Likely Vocoder: {vocoder_type}</h3>
            <div style="color: #cccccc;">
        """
        
        sorted_vocoders = sorted(vocoder_probs.items(), key=lambda x: x[1], reverse=True)
        for vocoder, prob in sorted_vocoders[:5]:
            bar_width = int(prob * 100)
            vocoder_html += f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span>{vocoder}</span>
                    <span>{prob:.1%}</span>
                </div>
                <div style="background: #333; border-radius: 5px; height: 20px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #ff6b6b, #ee5a6f); 
                         width: {bar_width}%; height: 100%; transition: width 0.3s;"></div>
                </div>
            </div>
            """
        
        vocoder_html += "</div></div>"
        
        progress(1.0, desc="Complete!")
        
        return verdict_html, fig1, fig2, vocoder_html
        
    except Exception as e:
        return f"❌ Error processing audio: {str(e)}", None, None, ""


# Custom CSS for dark theme
custom_css = """
.gradio-container {
    background: #0d0d0d !important;
    color: #ffffff !important;
}

.block {
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

.label {
    color: #ffffff !important;
}

button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: bold !important;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

.tabs {
    background: #1a1a1a !important;
}

.tab-nav {
    background: #0d0d0d !important;
    border-bottom: 2px solid #333333 !important;
}

.selected {
    background: #667eea !important;
}

h1, h2, h3 {
    color: #ffffff !important;
}

.footer {
    display: none !important;
}
"""


# Create Gradio app
with gr.Blocks(css=custom_css, title="🔍 Deepfake Detector", theme=gr.themes.Default(primary_hue="purple")) as app:
    
    # Header
    gr.Markdown("""
    # 🔍 Deepfake Detector
    ### Advanced GAN Fingerprint-Based Detection for Images, Videos, and Audio
    
    Upload media to detect deepfakes using multi-branch fusion neural networks. 
    The system analyzes spectral patterns, noise residuals, and temporal consistency to identify synthetic content.
    """)
    
    # Create tabs
    with gr.Tabs():
        
        # IMAGE DETECTION TAB
        with gr.TabItem("📸 Image Detection"):
            gr.Markdown("### Upload an image to detect if it contains a deepfake face")
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="filepath", label="Upload Image", height=400)
                    img_button = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Detected Face", height=400)
            
            with gr.Row():
                img_verdict = gr.HTML(label="Detection Result")
            
            with gr.Row():
                with gr.Column():
                    img_gan_type = gr.HTML(label="GAN Type Analysis")
                with gr.Column():
                    img_heatmap = gr.Image(label="GradCAM Heatmap")
            
            img_button.click(
                fn=detect_image,
                inputs=[img_input],
                outputs=[img_output, img_verdict, img_gan_type, img_heatmap]
            )
        
        # VIDEO DETECTION TAB
        with gr.TabItem("🎥 Video Detection"):
            gr.Markdown("### Upload a video to detect deepfake faces with temporal analysis")
            
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    vid_button = gr.Button("🔍 Analyze Video", variant="primary", size="lg")
            
            with gr.Row():
                vid_verdict = gr.HTML(label="Detection Result")
            
            with gr.Row():
                with gr.Column():
                    vid_timeline = gr.Plot(label="Per-Frame Timeline")
                with gr.Column():
                    vid_preview = gr.Image(label="Annotated Frame Preview")
            
            with gr.Row():
                vid_gan_type = gr.HTML(label="GAN Type Analysis")
            
            vid_button.click(
                fn=detect_video,
                inputs=[vid_input],
                outputs=[vid_verdict, vid_timeline, vid_preview, vid_gan_type]
            )
        
        # AUDIO DETECTION TAB
        with gr.TabItem("🎵 Audio Detection"):
            gr.Markdown("### Upload audio to detect synthetic voice and vocoders")
            
            with gr.Row():
                with gr.Column():
                    aud_input = gr.Audio(type="filepath", label="Upload Audio")
                    aud_button = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
            
            with gr.Row():
                aud_verdict = gr.HTML(label="Detection Result")
            
            with gr.Row():
                with gr.Column():
                    aud_segments = gr.Plot(label="Per-Segment Scores")
                with gr.Column():
                    aud_spectrogram = gr.Plot(label="Mel Spectrogram")
            
            with gr.Row():
                aud_vocoder = gr.HTML(label="Vocoder Type Analysis")
            
            aud_button.click(
                fn=detect_audio,
                inputs=[aud_input],
                outputs=[aud_verdict, aud_segments, aud_spectrogram, aud_vocoder]
            )
    
    # Footer
    gr.Markdown("""
    ---
    **Note:** This is a research prototype. Models are initialized with random weights if no checkpoint is provided.
    For production use, train the models on your dataset or load pre-trained checkpoints.
    
    **Supported Formats:**
    - Images: JPG, PNG
    - Videos: MP4, AVI
    - Audio: WAV, MP3, FLAC
    """)


if __name__ == "__main__":
    print("Starting Deepfake Detector Web UI...")
    print(f"Using device: {device}")
    print(f"Config loaded from: {CONFIG_PATH}")
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
