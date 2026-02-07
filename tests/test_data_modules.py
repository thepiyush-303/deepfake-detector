"""
Test suite for data preprocessing and dataset modules.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os


class TestPreprocessing:
    """Test image preprocessing module."""
    
    def test_extract_fingerprints(self):
        """Test fingerprint extraction from RGB image."""
        from data.preprocessing import extract_fingerprints
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        fingerprints = extract_fingerprints(dummy_image)
        
        # Check all required keys are present
        assert 'fft' in fingerprints
        assert 'dct' in fingerprints
        assert 'srm' in fingerprints
        assert 'rgb' in fingerprints
        
        # Check shapes
        assert fingerprints['fft'].shape == (256, 256)
        assert fingerprints['dct'].shape == (256, 256)
        assert fingerprints['srm'].shape == (30, 256, 256)
        assert fingerprints['rgb'].shape == (3, 256, 256)
        
        # Check dtypes
        assert fingerprints['fft'].dtype == np.float32
        assert fingerprints['dct'].dtype == np.float32
        assert fingerprints['srm'].dtype == np.float32
        assert fingerprints['rgb'].dtype == np.float32
    
    def test_batch_extract_fingerprints(self):
        """Test batch fingerprint extraction."""
        from data.preprocessing import batch_extract_fingerprints
        
        # Create batch of dummy images
        images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        results = batch_extract_fingerprints(images)
        
        assert len(results) == 3
        for fingerprints in results:
            assert fingerprints is not None
            assert 'fft' in fingerprints
            assert 'dct' in fingerprints
            assert 'srm' in fingerprints
            assert 'rgb' in fingerprints


class TestAudioPreprocessing:
    """Test audio preprocessing module."""
    
    def test_segment_audio(self):
        """Test audio segmentation."""
        from data.audio_preprocessing import segment_audio
        
        # Create synthetic waveform (5 seconds)
        sr = 16000
        waveform = torch.randn(1, sr * 5)
        
        segments = segment_audio(waveform, sr, segment_duration=4.0, hop_duration=2.0)
        
        assert len(segments) > 0
        assert segments[0].shape[0] == 1
        assert segments[0].shape[1] == sr * 4
    
    def test_compute_mel_spectrogram(self):
        """Test mel spectrogram computation."""
        from data.audio_preprocessing import compute_mel_spectrogram
        
        # Create synthetic waveform
        sr = 16000
        waveform = torch.randn(1, sr * 2)
        
        mel_spec = compute_mel_spectrogram(waveform, sr)
        
        assert mel_spec.shape[0] == 80  # n_mels
        assert mel_spec.dim() == 2
        assert mel_spec.dtype == torch.float32
    
    def test_compute_lfcc(self):
        """Test LFCC computation."""
        from data.audio_preprocessing import compute_lfcc
        
        # Create synthetic waveform
        sr = 16000
        waveform = torch.randn(1, sr * 2)
        
        lfcc = compute_lfcc(waveform, sr)
        
        assert lfcc.shape[0] == 40  # n_lfcc
        assert lfcc.dim() == 2
        assert lfcc.dtype == torch.float32


class TestDatasets:
    """Test Dataset classes."""
    
    def test_image_dataset(self):
        """Test DeepfakeImageDataset."""
        from data.dataset import DeepfakeImageDataset
        
        # Create dummy image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(dummy_image).save(f.name)
            image_path = f.name
        
        try:
            dataset = DeepfakeImageDataset(
                image_paths=[image_path, image_path],
                labels=[0, 1],
                gan_types=[0, 1]
            )
            
            assert len(dataset) == 2
            
            fingerprints, label, gan_type = dataset[0]
            
            assert isinstance(fingerprints, dict)
            assert 'fft' in fingerprints
            assert 'dct' in fingerprints
            assert 'srm' in fingerprints
            assert 'rgb' in fingerprints
            
            assert isinstance(label, (int, np.integer))
            assert isinstance(gan_type, (int, np.integer))
        finally:
            os.unlink(image_path)
    
    def test_image_dataset_dataloader(self):
        """Test DeepfakeImageDataset with DataLoader."""
        from data.dataset import DeepfakeImageDataset
        from torch.utils.data import DataLoader
        
        # Create dummy images
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(dummy_image).save(f.name)
            image_path = f.name
        
        try:
            dataset = DeepfakeImageDataset(
                image_paths=[image_path] * 4,
                labels=[0, 1, 0, 1],
                gan_types=[0, 1, 2, 3]
            )
            
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            
            for batch in loader:
                fingerprints, labels, gan_types = batch
                assert isinstance(fingerprints, dict)
                assert fingerprints['fft'].shape[0] == 2  # batch size
                break
        finally:
            os.unlink(image_path)


class TestAugmentation:
    """Test augmentation classes."""
    
    def test_visual_augmentation(self):
        """Test VisualAugmentation."""
        from data.augmentation import VisualAugmentation
        
        aug = VisualAugmentation()
        
        # Create dummy fingerprints
        fingerprints = {
            'fft': np.random.randn(256, 256).astype(np.float32),
            'dct': np.random.randn(256, 256).astype(np.float32),
            'srm': np.random.randn(30, 256, 256).astype(np.float32),
            'rgb': np.random.randn(3, 256, 256).astype(np.float32)
        }
        
        augmented = aug(fingerprints)
        
        assert 'fft' in augmented
        assert 'dct' in augmented
        assert 'srm' in augmented
        assert 'rgb' in augmented
    
    def test_audio_augmentation(self):
        """Test AudioAugmentation."""
        from data.augmentation import AudioAugmentation
        
        aug = AudioAugmentation()
        
        # Create dummy features
        features = torch.randn(80, 100)
        augmented = aug(features)
        
        assert augmented.shape == features.shape
        assert augmented.dtype == torch.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
