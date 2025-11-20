import torch
import torch.nn as nn

class Simple3DUNet(nn.Module):
    """Baseline 3D U-Net for diffusion models."""
    
    def __init__(self, in_channels=1, base_channels=16):
        super(Simple3DUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.dec3 = self._conv_block(base_channels * 12, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 6, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 3, base_channels)
        
        # Final output
        self.final_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder with skip connections
        dec3 = self.upsample(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

class Tiny3DUNet(nn.Module):
    """Optimized 3D U-Net with reduced architecture."""
    
    def __init__(self, in_channels=1, base_channels=8):
        super(Tiny3DUNet, self).__init__()
        
        # Simplified encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Simplified decoder
        self.dec2 = self._conv_block(base_channels * 6, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 3, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        
        bottleneck = self.bottleneck(self.pool(enc2))
        
        dec2 = self.upsample(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

class EdgeFusionUNet(nn.Module):
    """Highly optimized 3D U-Net for edge deployment."""
    
    def __init__(self, in_channels=1, base_channels=8):
        super(EdgeFusionUNet, self).__init__()
        
        # Minimal architecture
        self.enc1 = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv3d(base_channels, base_channels * 2, 3, padding=1)
        self.bottleneck = nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.dec2 = nn.Conv3d(base_channels * 6, base_channels * 2, 3, padding=1)
        self.dec1 = nn.Conv3d(base_channels * 3, base_channels, 3, padding=1)
        self.final_conv = nn.Conv3d(base_channels, in_channels, 1)
        
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        enc1 = self.relu(self.enc1(x))
        enc2 = self.relu(self.enc2(self.pool(enc1)))
        
        bottleneck = self.relu(self.bottleneck(self.pool(enc2)))
        
        dec2 = self.upsample(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.relu(self.dec2(dec2))
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.relu(self.dec1(dec1))
        
        return self.final_conv(dec1)
