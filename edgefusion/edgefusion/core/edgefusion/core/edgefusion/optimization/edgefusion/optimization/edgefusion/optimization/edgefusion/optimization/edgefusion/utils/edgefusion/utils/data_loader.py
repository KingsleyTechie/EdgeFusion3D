import torch
from torch.utils.data import Dataset
import numpy as np
import re

def load_off_file(file_path):
    """Load a .off file and return vertices and faces."""
    with open(file_path, 'r') as f:
        # Read the header
        header = f.readline().strip()
        if header != 'OFF' and header != 'COFF':
            parts = f.readline().split()
            num_vertices, num_faces, _ = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            parts = f.readline().split()
            num_vertices, num_faces, _ = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Read vertices
        vertices = []
        for _ in range(num_vertices):
            vertex_line = f.readline().split()
            if len(vertex_line) >= 3:
                vertices.append([float(vertex_line[0]), float(vertex_line[1]), float(vertex_line[2])])
        
        return np.array(vertices), []

def normalize_vertices(vertices):
    """Normalize vertices to fit in a unit cube centered at origin."""
    if len(vertices) == 0:
        return vertices
    
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = vertices - center
    
    max_extent = np.max(np.abs(vertices))
    if max_extent > 0:
        vertices = vertices / max_extent
    
    return vertices

def voxelize_vertices(vertices, grid_size=32):
    """Convert vertices to a voxel grid using point occupancy."""
    normalized_vertices = (vertices + 1) * 0.5
    voxel_coords = (normalized_vertices * (grid_size - 1)).astype(int)
    
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    for coord in voxel_coords:
        if (0 <= coord[0] < grid_size and 
            0 <= coord[1] < grid_size and 
            0 <= coord[2] < grid_size):
            voxel_grid[coord[0], coord[1], coord[2]] = 1.0
    
    return voxel_grid

class VoxelDataset(Dataset):
    def __init__(self, file_paths, grid_size=32, transform=None):
        self.file_paths = file_paths
        self.grid_size = grid_size
        self.transform = transform
        self.voxel_cache = {}
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if idx in self.voxel_cache:
            return self.voxel_cache[idx]
        
        file_path = self.file_paths[idx]
        
        try:
            vertices, faces = load_off_file(file_path)
            normalized_vertices = normalize_vertices(vertices)
            voxel_grid = voxelize_vertices(normalized_vertices, self.grid_size)
            
            voxel_tensor = torch.from_numpy(voxel_grid).unsqueeze(0)
            
            if self.transform:
                voxel_tensor = self.transform(voxel_tensor)
            
            self.voxel_cache[idx] = voxel_tensor
            return voxel_tensor
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((1, self.grid_size, self.grid_size, self.grid_size))
