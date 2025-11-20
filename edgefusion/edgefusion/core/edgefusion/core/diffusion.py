import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DiffusionTrainer:
    """Handles the diffusion training process for 3D voxel grids."""
    
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Pre-calculate diffusion parameters
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def forward_diffusion(self, x0, t):
        """Apply forward diffusion to sample x_t from x_0."""
        noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)
        
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise
    
    def train_step(self, x0, optimizer, loss_fn):
        """Perform a single training step."""
        self.model.train()
        optimizer.zero_grad()
        
        # Sample random timestep
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=self.device)
        
        # Apply forward diffusion
        x_t, noise = self.forward_diffusion(x0, t)
        
        # Predict noise
        predicted_noise = self.model(x_t)
        
        # Calculate loss
        loss = loss_fn(predicted_noise, noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def generate_sample(self, shape=(1, 32, 32, 32), num_steps=50):
        """Generate a sample using the trained diffusion model."""
        self.model.eval()
        
        # Start from random noise
        x = torch.randn((1, 1) + shape).to(self.device)
        
        # Reverse diffusion process
        for t in tqdm(range(num_steps-1, -1, -1), desc="Generating"):
            # Predict noise
            predicted_noise = self.model(x)
            
            # Remove noise (simplified reverse step)
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
                
            x = (x - beta_t * predicted_noise / torch.sqrt(1 - self.alpha_bars[t])) / torch.sqrt(alpha_t)
            x = x + torch.sqrt(beta_t) * z
        
        return x.cpu()
