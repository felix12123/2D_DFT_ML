import numpy as np
import torch
import torch.nn as nn

from ._layers import Swish, SphericalConv2d
from ._baseclass import BaseModel

# In FMT, the second convolution has the same weights as the first one, but flipped in both dimensions.
# In practice, it is easier to implement this by flipping the input tensor before and after the second convolution.
class Model_FMT(BaseModel):
    """Model that is strongly inspired by the FMT definition of c_1"""
    def __init__(self, Nd:int, Nn:int=16, hidden_channels:int=64, hidden_layers:int=6, convtype=nn.Conv2d, activation=Swish):
        """Initializes a Model_FMT object

        Args:
            Nd (int): Diameter of particle in bins
            Nn (int, optional): Number of weighted densities. Defaults to 16.
            hidden_channels (int, optional): hidden channels to use for \\phi Dense Network. Defaults to 128.
            hidden_layers (int, optional): Hidden layers to use for \\phi Dense Network. Defaults to 6.
            convtype (_type_, optional): Type of convolution to use. Defaults to nn.Conv2d.
        """
        super(Model_FMT, self).__init__()
        self.Nn = Nn
        self.Nd = Nd
        self.padsize = Nd
        self.omegas = convtype(Nn, Nn, 2*Nd+1, bias=False, groups=Nn)
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
        x = self.pad(x) # pad before convolution
        x = self.omegas(x) # Apply the omegas Convolution
        x = self.phi(x) # Apply the phi Dense Network
        x = torch.flip(x, [2, 3]) # We flip the profiles to simulate flipped convolution weights in the second convolution
        x = self.pad(x) # pad after flipping
        x = self.omegas(x) # Apply the omegas Convolution again
        x = torch.flip(x, [2, 3]) # Flip back to original order
        x = self.sum_channels(x) # Sum over the Nn channels to arrive at the disired output shape
        return x
    
    def collect_all_kernels(self):
        """
        Collects all kernels of the model and returns them as a list
        that contains a list for each Convolution layer that contains the kernels as numpy arrays.
        This is useful for visualisation purposes.
        """
        kernels = []
        for i in range(self.omegas.weight.shape[0]):
            kernel = self.omegas.weight[i].detach().cpu().numpy().copy()
            if isinstance(self.omegas, SphericalConv2d):
                self.omegas.apply_weight_constraint()
                kernel[kernel == 0] = np.nan
            kernels.append(kernel)
        return [kernels]