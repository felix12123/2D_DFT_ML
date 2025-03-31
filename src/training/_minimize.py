import torch
import os

import numpy as np
from ._simfolder_utils import get_Vext

from typing import TYPE_CHECKING
if TYPE_CHECKING: # pragma: no cover
    from . import MLTraining


def minimize(self:'MLTraining', Vext_func, mu=0, beta=1, max_iter=10000, dx=None, tol=1e-4, alpha=0.03, alpha_max=0.2, Lx=None, Ly=None, L=None):
    if L != None:
        Lx = L
        Ly = L
    self.model.add_padding()
    return _minimize(self.model, Vext_func, mu, beta, max_iter, dx, tol, alpha, alpha_max, Lx, Ly, self.device)

def _minimize(c1_func, Vext_func, mu=0, beta=1, max_iter=10000, dx=None, tol=1e-4, alpha=0.03, alpha_max=0.2, Lx=None, Ly=None, device=None, alpha_min=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if alpha_min is None:
        alpha_min = alpha / 100
    if (dx is None or Ly is None or Lx is None) and callable(Vext_func):
        raise ValueError("dx and L must be specified if Vext_func is a function")
    if dx is None:
        lx = Vext_func.shape[-2]
        ly = Vext_func.shape[-1]
    else:
        lx = int(Lx / dx)
        ly = int(Ly / dx)
    
    if callable(c1_func) and isinstance(c1_func, torch.nn.Module):
        shape = [1, 1, lx, ly]
    elif callable(c1_func):
        shape = [lx, ly]
    elif type(c1_func) == str:
        c1_func = torch.load(c1_func)
        shape = [1, 1, lx, ly]
    else:
        TypeError("c1_func must be a function or a torch.nn.Module")
        
    
    # get tensor from Vext_func
    if callable(Vext_func):
        xs = np.arange(dx/2, Lx, dx)
        ys = np.arange(dx/2, Ly, dx)
        Vext = torch.tensor([[Vext_func(xx, yy) for xx in xs] for yy in ys], dtype=torch.float32).reshape(*shape)
    elif type(Vext_func) == str:
        i = int(Vext_func.split("_")[-1].split(".")[0])
        Vext = torch.tensor(get_Vext(os.path.dirname(Vext_func), i), dtype=torch.float32).reshape(*shape)
        mu = 0
        beta = 1
    else:
        Vext = torch.from_numpy(np.float32(Vext_func)).reshape(*shape)
    
    inf_mask = torch.isinf(Vext)
    
    rho = torch.ones_like(Vext, dtype=torch.float32) * 0.5
    rho_el = torch.zeros_like(Vext, dtype=torch.float32)
    rho[inf_mask] = 0
    
    delta_rho_mean_old = 10 * tol
    delta_rho_mean = 0
    delta_rho_max = 0
    
    # move everything to device
    Vext = Vext.to(device)
    rho = rho.to(device)
    rho_el = rho_el.to(device)
    inf_mask = inf_mask.to(device)
    c1_func = c1_func.to(device)
    
    steps = 0
    with torch.no_grad():
        for i in range(max_iter):
            rho_el.copy_(torch.exp((mu - Vext) * beta + c1_func(rho)))
            rho.add_(-alpha * rho + alpha * rho_el)
            rho[inf_mask] = 0
            rho.clamp_(0, float('inf'))
            
            # mean squared error
            # delta_rho_max = torch.functional.F.mse_loss(rho, rho_el)
            delta_rho_mean = torch.mean(torch.abs(rho - rho_el)[~inf_mask])
            delta_rho_max = torch.max(torch.abs(rho - rho_el)[~inf_mask])
            print("Step %.4d, alpha = %.3e: Δρ = %.3e          \r" % (i, alpha, delta_rho_max), end='')
            if delta_rho_max < tol:
                steps = i
                print(f"Converged (step: {i}, ||Δρ|| = {delta_rho_max} < {tol} = tolerance)")
                break
            if alpha_max > 0:
                if delta_rho_mean > delta_rho_mean_old:
                    alpha = alpha / 2
                    alpha = max(alpha, alpha_min)
                else:
                    alpha = min(alpha * 1.1, alpha_max)
            delta_rho_mean_old = delta_rho_mean
            
            if i % 20 == 0:
                torch.cuda.empty_cache()
    
    if delta_rho_max > tol:
        print(f"Did not converge (step: {max_iter}, ||Δρ|| = {delta_rho_max} > {tol} = tolerance)")
        return rho, max_iter
    return rho, steps
    


    