import numpy as np
import json
import random
import math
import os

from src.training._simfolder_utils import get_Vext
from src.fmt import get_rho_func

def generate_random_potential(L, dx, N_sin, min_sin_amp, max_sin_amp, max_per, 
                             N_plat, max_plat_height, min_plat_size, max_plat_size,
                             min_plat_phi, max_plat_phi, wall_types, min_wall, max_wall,
                             min_plat_theta, max_plat_theta, min_mu, max_mu, beta=1, output_file=None):
    # Function to make a value divisible by dx
    def make_divisible_by_dx(value):
        return round(value / dx) * dx
    
    # Generate random parameters
    n_sin = random.randint(1, N_sin)
    n_plat = random.randint(0, N_plat)
    mu = random.uniform(min_mu, max_mu)
    
    # Generate sinusoidal parameters
    amplitudes = [random.uniform(min_sin_amp, max_sin_amp) for _ in range(n_sin)]
    phases = [random.uniform(0, 2*math.pi) for _ in range(n_sin)]
    
    # Generate non-zero integer periods
    periods_x = []
    periods_y = []
    for _ in range(n_sin):
        per_x = random.randint(-max_per, max_per)
        while per_x == 0:  # Ensure non-zero
            per_x = random.randint(-max_per, max_per)
        periods_x.append(per_x)
        
        per_y = random.randint(-max_per, max_per)
        while per_y == 0:  # Ensure non-zero
            per_y = random.randint(-max_per, max_per)
        periods_y.append(per_y)
    
    # Generate platform parameters
    plat_position_x = [make_divisible_by_dx(random.uniform(0, L)) for _ in range(n_plat)]
    plat_position_y = [make_divisible_by_dx(random.uniform(0, L)) for _ in range(n_plat)]
    
    plat_size_x = [make_divisible_by_dx(random.uniform(min_plat_size, max_plat_size)) for _ in range(n_plat)]
    plat_size_y = [make_divisible_by_dx(random.uniform(min_plat_size, max_plat_size)) for _ in range(n_plat)]
    
    plat_heights = [random.uniform(-max_plat_height, max_plat_height) for _ in range(n_plat)]
    plat_theta = [random.uniform(min_plat_theta, max_plat_theta) for _ in range(n_plat)]
    plat_phi = [random.uniform(min_plat_phi, max_plat_phi) for _ in range(n_plat)]
    
    # Generate wall parameters
    wall = random.choice(wall_types)
    wall_thickness = make_divisible_by_dx(random.uniform(min_wall, max_wall))
    
    # Create the potential dictionary
    potential = {
        "L": L,
        "dx": dx,
        "mu": mu,
        "beta": beta,
        "amplitudes": amplitudes,
        "phases": phases,
        "periods_x": periods_x,
        "periods_y": periods_y,
        "plat_position_x": plat_position_x,
        "plat_position_y": plat_position_y,
        "plat_size_x": plat_size_x,
        "plat_size_y": plat_size_y,
        "plat_heights": plat_heights,
        "plat_theta": plat_theta,
        "plat_phi": plat_phi,
        "wall": wall,
        "wall_thickness": wall_thickness
    }
    
    # Save to file if output_file is provided
    if output_file is None:
        output_file = f"potential_{random.randint(1, 1000)}.json"
    
    with open(output_file, 'w') as f:
        json.dump(potential, f, indent=4)
    
    return potential, output_file

def generate_multiple_potentials(num_files, output_dir="potentials", **kwargs):
    """
    Generate multiple potential files with random parameters.
    
    Parameters:
    -----------
    num_files : int
        Number of files to generate
    output_dir : str
        Directory to save the files
    **kwargs : 
        Parameters to pass to generate_random_potential function
    
    Returns:
    --------
    list
        List of generated filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    filenames = []
    for i in range(num_files):
        output_file = os.path.join(output_dir, f"potential_{i+1}.json")
        _, filename = generate_random_potential(output_file=output_file, **kwargs)
        filenames.append(filename)
    
    return filenames

# Example usage



def create_rhos(datafolder, L, dx, tol=1e-5):
    print("\tgetting fmt rho function...", end="")
    os.makedirs(datafolder + "/rho", exist_ok=True)
    os.makedirs(datafolder + "/unc", exist_ok=True)
    numpots = len(os.listdir(datafolder + "/pot"))
    Nx = int(np.ceil(L/dx))
    rho_func = get_rho_func(Nx, Nx, L, L, 0.5)
    print(" done.")
    print("\tstarting iterations... ")
    for i in range(1, numpots+1):
        print("\r\t", i, "/", numpots, end="") # Print progress
        Vext = get_Vext(datafolder, i)
    
        rho, error = rho_func(0, Vext, tol)
        error = np.abs(error)
        
        np.savetxt(datafolder + "/rho/rho_%d.csv" % i, rho, delimiter=',')
        np.savetxt(datafolder + "/unc/rhostd_%d.csv" % i, error, delimiter=',') # this is beeing saved as a placeholder for the uncertainty
    print()


def create_td(res_path="data/fmt_td", num_sys=10, tol=1e-6, L=10, dx=0.05, N_sin=6, min_sin_amp=0.1, max_sin_amp=0.5,
               max_per=2, N_plat=0, max_plat_height=1,
               min_plat_size=0.75, max_plat_size=3.0,
               min_plat_phi=0, max_plat_phi=math.pi/2,
               min_plat_theta=math.pi/6, max_plat_theta=math.pi/2,
               wall_types=["n"], min_wall=0.1, max_wall=1.0,
               min_mu=-1.0, max_mu=3.0):
    # set random seed for reproducibility
    random.seed(1)
    
    # Example parameters
    # params = {
    #     "L": 10.0,
    #     "dx": 0.05,
    #     "N_sin": 10,
    #     "min_sin_amp": 0.1,
    #     "max_sin_amp": 1.0,
    #     "max_per": 3,
    #     "N_plat": 3,
    #     "max_plat_height": 1,
    #     "min_plat_size": 0.75,
    #     "max_plat_size": 3.0,
    #     "min_plat_phi": 0,
    #     "max_plat_phi": math.pi/2,
    #     "min_plat_theta": math.pi/6,
    #     "max_plat_theta": math.pi/2,
    #     "wall_types": ["n", "b"],
    #     "min_wall": 0.1,
    #     "max_wall": 1.0,
    #     "min_mu": -1.0,
    #     "max_mu": 3.0
    # }
    params = {
        "L": L,
        "dx": dx,
        "N_sin": N_sin,
        "min_sin_amp": min_sin_amp,
        "max_sin_amp": max_sin_amp,
        "max_per": max_per,
        "N_plat": N_plat,
        "max_plat_height": max_plat_height,
        "min_plat_size": min_plat_size,
        "max_plat_size": max_plat_size,
        "min_plat_phi": min_plat_phi,
        "max_plat_phi": max_plat_phi,
        "min_plat_theta": min_plat_theta,
        "max_plat_theta": max_plat_theta,
        "wall_types": wall_types,
        "min_wall": min_wall,
        "max_wall": max_wall,
        "min_mu": min_mu,
        "max_mu": max_mu
    }
    
    
    # Generate multiple files
    print("creating random potentials...", end="")
    generate_multiple_potentials(num_sys, res_path + "/pot", **params)
    print("done.")
    print("creating rho files...")
    create_rhos(res_path, params["L"], params["dx"], tol=tol)
    print("done.")