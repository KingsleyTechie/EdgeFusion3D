import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_3d_voxels(voxel_grid, threshold=0.5, title="3D Voxels"):
    """Plot 3D voxel grid using plotly."""
    occupied_voxels = np.where(voxel_grid > threshold)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=occupied_voxels[0],
        y=occupied_voxels[1], 
        z=occupied_voxels[2],
        mode='markers',
        marker=dict(
            size=3,
            color=occupied_voxels[2],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        )
    )
    
    return fig

def create_performance_chart(metrics_dict, save_path=None):
    """Create performance comparison chart."""
    models = list(metrics_dict.keys())
    sizes = [metrics_dict[model]['size_mb'] for model in models]
    times = [metrics_dict[model]['inference_ms'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Size comparison
    bars1 = ax1.bar(models, sizes, color='skyblue')
    ax1.set_title('Model Size Comparison')
    ax1.set_ylabel('Size (MB)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Inference time comparison
    bars2 = ax2.bar(models, times, color='lightcoral')
    ax2.set_title('Inference Time Comparison')
    ax2.set_ylabel('Time (ms)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
