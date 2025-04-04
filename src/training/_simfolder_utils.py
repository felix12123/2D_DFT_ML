import os
import numpy as np
from scipy.signal import convolve2d
from numpy.lib.stride_tricks import sliding_window_view


from ._pot import load_potential_params, evaluate_potential
import numpy as np

def check_df_health(folder:str):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a directory")
    if not os.access(folder, os.R_OK):
        raise PermissionError(f"Folder {folder} is not readable")
    if not os.access(folder, os.W_OK):
        raise PermissionError(f"Folder {folder} is not writable")
    if not os.access(folder, os.X_OK):
        raise PermissionError(f"Folder {folder} is not executable")
    
    # check for necessary subfolders
    if not os.path.isdir(folder + "/unc"):
        raise NotADirectoryError(f"folder {folder}/unc does not exist")
    if not os.path.isdir(folder + "/rho"):
        raise NotADirectoryError(f"folder {folder}/rho does not exist")
    if not os.path.isdir(folder + "/pot"):
        raise NotADirectoryError(f"folder {folder}/pot does not exist")
    
    num_files = max_file_num(folder)
    # check if all potential files are present
    for i in range(1, 1+num_files):
        if not os.path.isfile(folder + "/pot/potential_" + str(i) + ".json"):
            raise FileNotFoundError(f"potential_{i}.json not found")
    # check if all rho files are present
    for i in range(1, 1+num_files):
        if not os.path.isfile(folder + "/rho/rho_" + str(i) + ".csv"):
            raise FileNotFoundError(f"rho_{i}.csv not found")
    
    for i in range(1, 1+num_files):
        if not os.path.isfile(folder + "/unc/rhostd_" + str(i) + ".csv"):
            raise FileNotFoundError(f"rhostd_{i}.csv not found")
    
    # check if data in rho and unc is square and of same size
    rho_size = np.loadtxt(folder + "/rho/rho_1.csv", delimiter=",").shape
    unc_size = np.loadtxt(folder + "/unc/rhostd_1.csv", delimiter=",").shape
    if rho_size != unc_size:
        raise ValueError(f"rho and unc files are not of the same size")
    if rho_size[0] != rho_size[1]:
        raise ValueError(f"rho and unc files are not square")
    

def vectorized_2d_integration(Vext_large, l, n, dx_large):
    """
    Performs vectorized 2D integration on a large grid divided into l×l cells.
    
    Parameters:
    -----------
    Vext_large : ndarray
        The large grid of potential values
    l : int
        Number of cells in each dimension
    n : int
        Number of subdivisions within each cell
    dx_large : float
        Grid spacing in the large grid
        
    Returns:
    --------
    ndarray
        Integrated values for each cell with shape (l, l)
    """
    # Initialize output array
    integrated_values = np.zeros((l, l))
    
    # For each cell, apply the trapezoidal rule with appropriate weights
    for i in range(l):
        for j in range(l):
            # Extract the subgrid for this cell
            cell = Vext_large[i*n:(i+1)*n+1, j*n:(j+1)*n+1]
            
            # Interior points (full weight)
            interior_sum = np.sum(cell[1:-1, 1:-1])
            
            # Edge points (half weight)
            edge_sum = (np.sum(cell[0, 1:-1]) + 
                        np.sum(cell[-1, 1:-1]) + 
                        np.sum(cell[1:-1, 0]) + 
                        np.sum(cell[1:-1, -1])) * 0.5
            
            # Corner points (quarter weight)
            corner_sum = (cell[0, 0] + cell[0, -1] + 
                          cell[-1, 0] + cell[-1, -1]) * 0.25
            
            # Calculate the integral for this cell
            integrated_values[i, j] = (interior_sum + edge_sum + corner_sum) * dx_large * dx_large
    
    return integrated_values

def mean_integrator(Vext_large, l, n, dx_large):
    """
    Calculate the mean value of each cell in the Vext_large grid.
    
    Parameters:
    -----------
    Vext_large : ndarray
        The large grid of potential values
    l : int
        Number of cells in each dimension
    n : int
        Number of subdivisions within each cell
    dx_large : float
        Grid spacing in the large grid (not used for mean calculation)
        
    Returns:
    --------
    ndarray
        Mean values for each cell with shape (l, l)
    """
    # Create a sliding window view of the large grid with shape (l, l, n, n)
    windows = sliding_window_view(Vext_large, (n, n))[::n, ::n, :, :]
    
    infs = np.isinf(windows)
    inf_count = np.sum(infs, axis=(2, 3))

    # Calculate the mean of each window
    with np.errstate(divide = 'ignore'):
        # res = np.mean(windows, axis=(2, 3))
        res = -np.log(np.mean(np.exp(-windows), axis=(2, 3)))
        res[inf_count > 4*n] = np.inf
    return res
    
def calculate_Vext(datafolder:str, i:int, n:int=7):
    n = int(n)
    if n < 1:
        raise ValueError("n must be greater than or equal to 1")
    
    # Load the potential parameters
    pot_file = datafolder + "/pot/potential_" + str(i) + ".json"
    params = load_potential_params(pot_file)
    L = params["L"]
    dx = params["dx"]
    
    l = int(np.ceil((L/dx)))
    dx_large = L/(l*n) # dx of the large grid
    
    xs = np.linspace(dx_large/2, L-dx_large/2, l*n)
    X, Y = np.meshgrid(xs, xs)
    
    params_large = params.copy()
    params_large["dx"] = dx_large
    
    Vext_large = evaluate_potential(X, Y, params_large)
    Vext = mean_integrator(Vext_large, l, n, dx_large)
    
    return Vext - params["mu"]

def get_Vext(folder:str, i:int, n:int=7)->np.ndarray:
    """returns the effective potential for the i-th potential in the folder. This includes mu.

    Args:
        folder (str): base folder of the training data
        i (int): index of the potential. starts at 1
        n (int, optional): number of points to use for averaging in each direction. Defaults to 7. This means, that n*n points are used for each cell.

    Returns:
        np.ndarray: the effective potential
    """
    if os.path.exists(f"{folder}/Vext/Vext_{i}.csv"):
        Vext = np.loadtxt(f"{folder}/Vext/Vext_{i}.csv", delimiter=",")
    else:
        Vext = calculate_Vext(folder, i, n)
        if not os.path.exists(f"{folder}/Vext"):
            os.makedirs(f"{folder}/Vext")
        np.savetxt(f"{folder}/Vext/Vext_{i}.csv", Vext, delimiter=",")
    return Vext


def get_bmloc(potfolder: str, i: int):
    pot_file = potfolder + "/potential_" + str(i) + ".json"
    data = load_potential_params(pot_file)
    return -data["beta"] *  get_Vext(potfolder, i)



def get_rho(folder:str, i:int)->np.ndarray:
    return np.loadtxt(f"{folder}/rho/rho_{i}.csv", delimiter=",")

def get_rho_unc(folder:str, i:int)->np.ndarray:
    return np.loadtxt(f"{folder}/unc/rhostd_{i}.csv", delimiter=",")

def avg_c1_std(rho, rho_std)->float:
    return np.mean(rho_std[rho>0] / rho[rho>0])

def get_c1(folder:str, i:int, rho=None, Vext=None)->np.ndarray:
    if Vext is None:
        Vext = get_Vext(folder, i)
    if rho is None:
        rho = get_rho(folder, i)
    else:
        assert rho.shape == Vext.shape
    beta = load_potential_params(folder + f"/pot/potential_{i}.json")["beta"]
    c1 = np.zeros_like(rho)
    c1[rho > 0] = np.log(rho[rho > 0]) + beta * Vext[rho > 0]
    
    return c1


def max_file_num(res_folder:str)->int:
    """Returns the maximum index of files in a datafolder with the format 'rho_{num}.csv'"""
    rho_folder = res_folder + "/rho"
    if not os.path.exists(rho_folder):
        # warn
        print(f"Folder {rho_folder} does not exist")
        return 0
    files = os.listdir(rho_folder)
    indices = [int(file.split("_")[-1].split(".")[0]) for file in files if file.endswith(".csv")]
    return max(indices) if indices else 0
