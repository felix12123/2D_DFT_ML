import os
import numpy as np
from src.fmt import get_c1_and_rho_func
from src.training._plotting import plot_g_r

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.training import MLTraining


def _avg_over_intervals(dist, rho, dx):
    # sort by distance
    dist, rho = zip(*sorted(zip(dist, rho)))
    
    distances_avg = np.linspace(dx/2, max(dist), int(np.ceil((max(dist)-dx/2) / dx)) + 1)
    rho_avg = np.zeros(len(distances_avg))
    
    
    idx = 0
    idx_old = 0
    for i, d in enumerate(distances_avg):
        # find first rho value that is larger than d + 0.5dx
        while dist[idx] < d+0.5*dx:
            if idx == len(dist)-1:
                break
            idx += 1
        if idx == idx_old:
            continue
        rho_avg[i] = np.mean(rho[idx_old:idx])
        idx_old = idx
    return distances_avg, rho_avg
        
def rho_to_g_r(rho, dx, L, bins_dx=None):
    if bins_dx is None:
        bins_dx = dx
    distance = []
    rhos = []
    xs = np.linspace(0, L, int(round(L//dx)))
    ys = np.linspace(0, L, int(round(L//dx)))
    N = len(xs)
    
    for i in range(N):
        for j in range(N):
            dist = np.sqrt((xs[i]-L/2)**2 + (ys[j]-L/2)**2)
            if dist >= 1:
                distance.append(dist)
                rhos.append(rho[i, j])

    return _avg_over_intervals(distance, rhos, bins_dx)
    
def g_r(self: 'MLTraining', mus=None, plot_save_path=None, eps=1e-6):
    if not mus:
        mus = [0, 1, 2, 3, 4]
    if not plot_save_path:
        plot_save_path = self.workspace + "/plots/g_r_vs_FMT.pdf"
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    
    def Vext(x, y): # potential function
        if ((x-self.L/2)**2 + (y-self.L/2)**2 < 1):
            return np.inf
        else:
            return 0
    def Vext_vect(x, y): # vectorized potential function
        dist_squared = (x - self.L/2)**2 + (y - self.L/2)**2
        V = np.where(dist_squared < 1, np.inf, 0)
        return V
        
    N = int(round(self.L/self.dx))
    
    rho_and_c1_func = get_c1_and_rho_func(N, N, self.L, self.L, 0.5)

    rhos = []
    distances = []
    rhos_fmt = []
    distances_fmt = []
    for mu in mus:
        rho_min = self.minimize(Vext, mu, 1, 750, self.dx, L=self.L, tol=eps, alpha_max=0.5)[0].cpu().numpy().flatten().reshape(N, N)
        rho_fmt_min = rho_and_c1_func(mu, Vext_vect, eps)[0]
        
        distances_avg, rho_avg = rho_to_g_r(rho_min, self.dx, self.L)
        rhos.append(rho_avg)
        distances.append(distances_avg)
        
        distances_avg_fmt, rho_avg_fmt = rho_to_g_r(rho_fmt_min, self.dx, self.L)
        rhos_fmt.append(rho_avg_fmt)
        distances_fmt.append(distances_avg_fmt)
    
    fig = plot_g_r(distances, rhos, rhos_fmt, "FMT", [r"$\mu = " + str(mu) + "$" for mu in mus], save_path=plot_save_path)
    return fig










