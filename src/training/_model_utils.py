import torch
from torch import nn
from scipy.ndimage import gaussian_filter

   

def smooth_kernel(self, sigma):
    assert isinstance(self.model, nn.Module)
    
    with torch.no_grad():
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1:
                weights = m.weight.data.cpu().numpy()
                for i in range(weights.shape[0]):
                    for j in range(weights.shape[1]):
                        weights[i, j] = gaussian_filter(weights[i, j], sigma=sigma)
                m.weight.data.copy_(torch.tensor(weights).to(m.weight.device))
                if hasattr(self.model, 'apply_weight_constraint'):
                    self.model.apply_weight_constraint()


