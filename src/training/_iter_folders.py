# ÜBERARBEITET
import os
import time
import numpy as np

from .. import fmt as fmt
from ._simfolder_utils import get_Vext, check_df_health, max_file_num, get_rho

from typing import TYPE_CHECKING
if TYPE_CHECKING: # pragma: no cover
    from . import MLTraining

def fmt_profiles_exist(simfolder, fmt_folder:str=None)->bool:
    if fmt_folder is None:
        fmt_folder = simfolder + "/FMT"
    if not os.path.exists(fmt_folder):
        return False
    
    num_sim_profiles = max_file_num(simfolder)
    num_fmt_profiles = max_file_num(fmt_folder)
    if num_sim_profiles != num_fmt_profiles:
        return False
    
    return True

# calculate FMT profiles for a given folder and save them in a subfolder
def create_FMT_profiles(simfolder:str, eps:float=1e-5, L:float=10, savefolder=None, Vext_integration_n=7):
    if not os.path.exists(simfolder):
        raise FileNotFoundError(f"Folder '{simfolder}' not found")
    if savefolder is None:
        savefolder = simfolder + "/FMT/rho"
    
    check_df_health(simfolder) # check if all files are present
    
    print(f"Creating FMT profiles in folder '{simfolder}'")
    
    os.makedirs(savefolder, exist_ok=True)
    
    Nx = get_rho(simfolder, 1).shape[0]
    Ny = Nx
    R = 0.5
    
    c1_and_rho_func = fmt.get_c1_and_rho_func(Nx, Ny, L, L, R)

    
    print("starting comparison...")
    t0 = time.time()
    num_profiles = max_file_num(simfolder)
    for i in range(1, 1+num_profiles):
        print(i, "/", num_profiles, end="")
        if i > 1: print("...   (time left: %.2f min)" % ((num_profiles-i+1) * (time.time() - t0)/60/(i-0.999)))
        print("\r", end="")
        
        Vext = get_Vext(simfolder, i, Vext_integration_n)
        if Vext.min() < -12:
            print(f"Skipped. Vext min: {Vext.min()}")
            continue
        rho_min = c1_and_rho_func(0, Vext, eps)[0]
        # save the FMT profile
        np.savetxt(savefolder + f"/rho_{i}.csv", rho_min, delimiter=",")
    print("finished!")


def create_model_iteration_profiles(self:'MLTraining', potfolder:str=None, ml_iter_folder:str=None, tol:float=1e-6, L:float=10, max_iter=500, Vext_integration_n=7):
    """Create model iteration profiles for all profiles in the datafolder"""
    if potfolder is None:
        potfolder = self.datafolder
    if ml_iter_folder is None:
        ml_iter_folder = self.workspace + "/iters/" + potfolder.split("/")[-1]
    os.makedirs(ml_iter_folder + "/rho", exist_ok=True)
    
    check_df_health(potfolder) # check if all files are present
    
    print(f"Creating model iteration profiles in folder '{ml_iter_folder}'")
    
    indexes = np.array(range(1, 1+max_file_num(potfolder)))
    # np.random.shuffle(iss)
    for i in indexes:
        print(i, "/", max_file_num(potfolder), end="")
        print("\r", end="")
        
        Vext = get_Vext(potfolder, i, Vext_integration_n)
        rho_min = self.minimize(Vext, tol=tol, L=L, max_iter=max_iter)[0].cpu().numpy().squeeze()
        
        # save the profile
        np.savetxt(ml_iter_folder + f"/rho/rho_{i}.csv", rho_min, delimiter=",")



def ensure_model_iter_existance(self, potfolder=None, ml_iter_savefolder=None, eps=1e-5, max_iter=500, Vext_integration_n=7):
    """Ensures that the model iteration profiles exist for all profiles in the datafolder"""
    if potfolder is None:
        potfolder = self.datafolder
    if ml_iter_savefolder is None:
        ml_iter_savefolder = self.workspace + "/iters/" + potfolder.split("/")[-1]
    
    if max_file_num(ml_iter_savefolder) < max_file_num(potfolder):
        create_model_iteration_profiles(self, potfolder=potfolder, ml_iter_folder=ml_iter_savefolder, tol=eps, max_iter=max_iter, Vext_integration_n=Vext_integration_n)

def ensure_fmt_iter_existance(self, potfolder=None, fmt_savefolder=None, eps=1e-5, L=10):
    """Ensures that the FMT profiles exist for all profiles in the datafolder"""
    if potfolder is None:
        potfolder = self.datafolder
    if fmt_savefolder is None:
        fmt_savefolder = potfolder + "/FMT"
    
    if max_file_num(fmt_savefolder) < max_file_num(potfolder):
        create_FMT_profiles(potfolder, eps=eps, L=L, savefolder=fmt_savefolder)