import numpy as np
import torch.nn as nn
import torch

from ._layers import Swish, SphericalConv2d
from ._baseclass import BaseModel


class Model_FMT2(BaseModel):
    """Model that is lightly inspired by the FMT definition of c_1. In contrast to Model_FMT its two convolutions are not shared."""
    def __init__(self, Nd, Nn=16, hidden_channels=64, hidden_layers=6, convtype=nn.Conv2d, activation=Swish):
        """Initializes a Model_FMT2 object

        Args:
            Nd (int): Diameter of particle in bins
            Nn (int, optional): Number of weighted densities. Defaults to 16.
            hidden_channels (int, optional): hidden channels to use for \\phi Dense Network. Defaults to 128.
            hidden_layers (int, optional): Hidden layers to use for \\phi Dense Network. Defaults to 6.
            convtype (_type_, optional): Type of convolution to use. Defaults to nn.Conv2d.
        """
        super(Model_FMT2, self).__init__()
        self.Nn = Nn
        self.Nd = Nd
        self.padsize = Nd
        self.omegas1 = convtype(Nn, Nn, 2*Nd+1, bias=False, groups=Nn)
        self.omegas2 = convtype(Nn, Nn, 2*Nd+1, bias=False, groups=Nn)
        self.pad = nn.CircularPad2d(self.padsize)
        self.sum_channels = nn.Conv2d(Nn, 1, 1, padding=0, bias=False)
        seq_components = []
        channels = [Nn] + [hidden_channels] * (hidden_layers-1) + [Nn]
        for i in range(hidden_layers):
            seq_components.append(nn.Conv2d(channels[i], channels[i+1], 1))
            seq_components.append(activation())
        self.phi = nn.Sequential(*seq_components)
        
    def forward(self, x):
        # Assuming x has the shape [N, 1, L, L]
        x = x.repeat(1, self.Nn, 1, 1)
        x = self.pad(x)
        x = self.omegas1(x)
        x = self.phi(x)
        x = torch.flip(x, [2, 3])
        x = self.pad(x)
        x = self.omegas2(x)
        x = torch.flip(x, [2, 3])
        x = self.sum_channels(x)
        return x
    
    def collect_all_kernels(self):
        kernels1 = []
        for i in range(self.omegas1.weight.shape[0]):
            kernel = self.omegas1.weight[i].detach().cpu().numpy().copy()
            if isinstance(self.omegas1, SphericalConv2d):
                self.omegas1.apply_weight_constraint()
                kernel[kernel == 0] = np.nan
            kernels1.append(kernel)
        kernels2 = []
        for i in range(self.omegas2.weight.shape[0]):
            kernel = self.omegas2.weight[i].detach().cpu().numpy().copy()
            if isinstance(self.omegas2, SphericalConv2d):
                self.omegas2.apply_weight_constraint()
                kernel[kernel == 0] = np.nan
            kernels2.append(kernel)
        return [kernels1, kernels2]