import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class SphericalConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernelsize = self.weight.shape[2]  # Assuming square kernel
        self.apply_weight_constraint()
        self.register_hook()

    def register_hook(self):
        def hook_fn(grad):
            mask = self.get_circular_mask()
            grad *= mask  # Apply the mask to gradients
            return grad
        
        self.weight.register_hook(hook_fn)

    def apply_weight_constraint(self):
        """Applies the circular mask to the weights only at initialization."""
        with torch.no_grad():
            mask = self.get_circular_mask()
            self.weight *= mask

    def get_circular_mask(self):
        """Creates a mask that keeps weights inside a centered circle and zeros out others."""
        radius = self.kernelsize // 2  # Define the circular region
        y, x = np.ogrid[:self.kernelsize, :self.kernelsize]
        center = (radius, radius)
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        return torch.tensor(mask, dtype=self.weight.dtype, device=self.weight.device).view(1, 1, self.kernelsize, self.kernelsize)

    def forward(self, x):
        """Ensures weight constraint is applied before forward pass."""
        return super().forward(x)