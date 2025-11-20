import torch
import torch.nn as nn

def simulate_quantization(model, bits=8):
    """
    Simulate quantization by reducing precision.
    
    Args:
        model: PyTorch model to quantize
        bits: Target bit precision
    
    Returns:
        Quantized model
    """
    scale = 2**(bits - 1) - 1
    
    with torch.no_grad():
        for param in model.parameters():
            if param.data.dtype == torch.float32:
                # Simulate quantization noise
                quantized = torch.round(param.data * scale) / scale
                param.data = quantized
    
    return model

def measure_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
