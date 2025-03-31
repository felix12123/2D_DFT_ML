import torch.nn as nn
from ._layers import SphericalConv2d

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def remove_padding(self):
        self.pad = nn.Identity()
        
    def is_padding_zero(self):
        return isinstance(self.pad, nn.Identity)
    
    def add_padding(self):
        if not hasattr(self, 'padsize'):
            self.padsize = self.Nd
        self.pad = nn.CircularPad2d(self.padsize)
    
    def apply_weight_constraint(self):
        for m in self.modules():
            if isinstance(m, SphericalConv2d):
                m.apply_weight_constraint()