"""
EdgeFusion: A framework for optimized 3D diffusion models on edge devices.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from edgefusion.core.diffusion import DiffusionTrainer
from edgefusion.core.unet_3d import Simple3DUNet, Tiny3DUNet, EdgeFusionUNet
from edgefusion.optimization.pruning import apply_manual_pruning
from edgefusion.optimization.quantization import simulate_quantization

__all__ = [
    "DiffusionTrainer",
    "Simple3DUNet", 
    "Tiny3DUNet",
    "EdgeFusionUNet",
    "apply_manual_pruning",
    "simulate_quantization"
]
