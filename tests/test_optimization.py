import torch
import pytest
from edgefusion.core.unet_3d import Simple3DUNet
from edgefusion.optimization import apply_manual_pruning, simulate_quantization

class TestOptimization:
    def test_pruning(self):
        model = Simple3DUNet(in_channels=1, base_channels=8)
        initial_params = sum(p.numel() for p in model.parameters())
        
        pruned_model, stats = apply_manual_pruning(model, pruning_threshold=0.1)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        assert stats['pruned_parameters'] > 0
        assert initial_params == pruned_params  # Parameters zeroed but not removed
    
    def test_quantization(self):
        model = Simple3DUNet(in_channels=1, base_channels=8)
        quantized_model = simulate_quantization(model, bits=8)
        
        # Test that model still works
        x = torch.randn(1, 1, 32, 32, 32)
        output = quantized_model(x)
        assert output.shape == x.shape
