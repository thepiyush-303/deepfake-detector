"""Models module initialization."""

# Model architectures
from .srm_kernels import get_srm_kernels, SRMConv2d
from .spectrum_branch import SpectrumBranch
from .noise_branch import NoiseBranch
from .rgb_branch import RGBBranch
from .fusion_model import DeepfakeFusionModel
from .audio_model import AudioDeepfakeModel

__all__ = [
    'get_srm_kernels',
    'SRMConv2d',
    'SpectrumBranch',
    'NoiseBranch',
    'RGBBranch',
    'DeepfakeFusionModel',
    'AudioDeepfakeModel',
]

