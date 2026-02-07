"""
GradCAM visualization for model explainability.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from models.fusion_model import DeepfakeFusionModel
from data.preprocessing import extract_fingerprints


class GradCAM:
    """
    GradCAM (Gradient-weighted Class Activation Mapping) for visual explainability.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.
        
        Args:
            model: Neural network model
            target_layer: Target layer for activation extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate GradCAM heatmap.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for predicted class)
        
        Returns:
            GradCAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        
        # Create one-hot encoded target
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        
        # Backward
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def generate_gradcam(model, image, target_class=None, branch='spectrum'):
    """
    Generate GradCAM heatmap for a specific branch.
    
    Args:
        model: DeepfakeFusionModel instance
        image: RGB image as numpy array (H, W, 3)
        target_class: Target class (0 for real, 1 for fake)
        branch: Branch to visualize ('spectrum', 'noise', or 'rgb')
    
    Returns:
        Dictionary containing:
            - heatmap: GradCAM heatmap (H, W)
            - overlay: Heatmap overlayed on original image
            - prediction: Model prediction
    """
    device = next(model.parameters()).device
    
    # Extract fingerprints
    fingerprints = extract_fingerprints(image)
    
    # Prepare inputs
    rgb = torch.from_numpy(fingerprints['rgb']).unsqueeze(0).float().to(device)
    spectrum = np.stack([fingerprints['fft'], fingerprints['dct']], axis=0)
    spectrum = torch.from_numpy(spectrum).unsqueeze(0).float().to(device)
    noise = torch.from_numpy(fingerprints['srm']).unsqueeze(0).float().to(device)
    
    # Store for hooks
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    
    # Select target layer and input based on branch
    if branch == 'spectrum':
        target_layer = model.spectrum_branch.backbone.blocks[-1][-1]  # Last layer in last block
        branch_model = model.spectrum_branch
        branch_input = spectrum
    elif branch == 'noise':
        target_layer = model.noise_branch.backbone.blocks[-1][-1]
        branch_model = model.noise_branch
        branch_input = noise
    elif branch == 'rgb':
        target_layer = model.rgb_branch.backbone.blocks[-1][-1]
        branch_model = model.rgb_branch
        branch_input = rgb
    else:
        raise ValueError(f"Unknown branch: {branch}")
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass through the branch
        model.train()  # Need to be in train mode for gradients
        branch_model.train()
        
        features = branch_model(branch_input)
        
        # We need to create a scalar output for backward
        # Use the mean of features as the target
        output = features.mean()
        
        # Backward pass
        model.zero_grad()
        output.backward()
        
        # Generate heatmap from activations and gradients
        if activations is not None and gradients is not None:
            # Global average pooling on gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            
            # Weighted combination of activations
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize to [0, 1]
            cam = cam.squeeze().detach().cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
        else:
            # Fallback: create a dummy heatmap
            cam = np.ones((8, 8)) * 0.5
    
    finally:
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        model.eval()
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        binary_probs, gan_type_probs, _ = model(rgb, spectrum, noise, return_probs=True)
    
    fakeness_score = binary_probs.squeeze().cpu().item()
    prediction = 'FAKE' if fakeness_score > 0.5 else 'REAL'
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Create overlay
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    return {
        'heatmap': heatmap_resized,
        'overlay': overlay,
        'prediction': prediction,
        'fakeness_score': fakeness_score
    }


def visualize_all_branches(image, model):
    """
    Create combined visualization showing GradCAM for all branches.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        model: DeepfakeFusionModel instance
    
    Returns:
        Matplotlib figure
    """
    # Generate GradCAM for each branch
    spectrum_gradcam = generate_gradcam(model, image, branch='spectrum')
    noise_gradcam = generate_gradcam(model, image, branch='noise')
    rgb_gradcam = generate_gradcam(model, image, branch='rgb')
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    
    # Original image
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Spectrum branch heatmap
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(spectrum_gradcam['heatmap'], cmap='jet')
    ax2.set_title('Spectrum Branch Heatmap', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Spectrum branch overlay
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(spectrum_gradcam['overlay'])
    ax3.set_title('Spectrum Branch Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Prediction
    ax4 = plt.subplot(2, 4, 4)
    ax4.axis('off')
    
    prediction = spectrum_gradcam['prediction']
    fakeness_score = spectrum_gradcam['fakeness_score']
    
    pred_color = 'red' if prediction == 'FAKE' else 'green'
    pred_text = f"Prediction: {prediction}\n\n"
    pred_text += f"Fakeness Score: {fakeness_score:.4f}\n"
    pred_text += f"Realness Score: {1.0 - fakeness_score:.4f}"
    
    ax4.text(0.5, 0.5, pred_text,
             fontsize=14, fontweight='bold',
             ha='center', va='center',
             color=pred_color,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Noise branch heatmap
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(noise_gradcam['heatmap'], cmap='jet')
    ax5.set_title('Noise Branch Heatmap', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Noise branch overlay
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(noise_gradcam['overlay'])
    ax6.set_title('Noise Branch Overlay', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # RGB branch heatmap
    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(rgb_gradcam['heatmap'], cmap='jet')
    ax7.set_title('RGB Branch Heatmap', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # RGB branch overlay
    ax8 = plt.subplot(2, 4, 8)
    ax8.imshow(rgb_gradcam['overlay'])
    ax8.set_title('RGB Branch Overlay', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    plt.tight_layout()
    return fig
