import torch
import torch.nn as nn

def apply_manual_pruning(model, pruning_threshold=0.01):
    """
    Apply manual pruning by zeroing out small weights.
    
    Args:
        model: PyTorch model to prune
        pruning_threshold: Magnitude threshold for pruning
    
    Returns:
        Pruned model and statistics
    """
    pruned_count = 0
    total_count = 0
    
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) > 1:  # Only prune weight matrices, not biases
                mask = torch.abs(param) > pruning_threshold
                pruned_count += torch.sum(~mask).item()
                total_count += param.numel()
                param.data *= mask.float()
    
    pruning_stats = {
        'pruned_parameters': pruned_count,
        'total_parameters': total_count,
        'pruning_ratio': pruned_count / total_count if total_count > 0 else 0
    }
    
    return model, pruning_stats
