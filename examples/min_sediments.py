from src.training import *
from matplotlib import pyplot as plt
import torch
import numpy as np

# define potential matrix

Lx = 16
Ly = 64*4
dx = 0.05

xs = np.arange(dx/2, Lx, dx)
ys = np.arange(dx/2, Ly, dx)
Vext = np.array([[(yy-1) * 5/Ly for xx in xs] for yy in ys])
# wall at the bottom from x=0 to x=2
Vext[0, 0:2*int(1/dx)] = np.inf

Vext1d = np.mean(Vext, 1)
plt.plot(ys, Vext1d)
plt.savefig("Vext.png")


mlt = load_MLTraining("saved_trainings/mlt_dx005-mixed-2_Model_FMT2_Nn32.pt")
# mlt = load_MLTraining("saved_trainings/mlt_FMT_Model_FMT2_Nn32.pt")

rho = mlt.minimize(Vext, mu=4, dx=dx, Lx=Lx, Ly=Ly, device="cuda", max_iter=200, alpha=0.1, alpha_max=0.2, tol=1e-6)[0].cpu().squeeze().numpy()




def plot_density_1D(rho, path:str, yscale="linear"):
    if len(rho.shape) != 1:
        raise ValueError("rho must be 1D")
    plt.figure(figsize=(5, 5), facecolor=(1,1,1,0))
    plt.plot(rho)
    plt.yscale(yscale)
    plt.savefig(path)
    plt.close()

# calculate projected 1d density along x axis
print("shape of rho: ", rho.shape)
plot_density_1D(np.mean(rho, 1), "rho.png")
# plot_density_1D(np.mean(rho, axis=2), "rho.png")

