from edgefusion.core.unet_3d import Tiny3DUNet, EdgeFusionUNet

def create_optimized_architecture(architecture_type='tiny', **kwargs):
    """
    Create optimized model architecture.
    
    Args:
        architecture_type: Type of architecture ('tiny' or 'edgefusion')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Optimized model instance
    """
    if architecture_type == 'tiny':
        return Tiny3DUNet(**kwargs)
    elif architecture_type == 'edgefusion':
        return EdgeFusionUNet(**kwargs)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
