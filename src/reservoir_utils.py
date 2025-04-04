import os
import numpy as np
from src.training._simfolder_utils import max_file_num, get_rho, get_rho_unc, get_c1, check_df_health, get_Vext

import matplotlib.pyplot as plt


def plot_histograms(data, name, filepath):
    plt.figure(figsize=(4, 3), dpi=300, facecolor=(1,1,1,0))
    plt.hist(data, bins=20, edgecolor="black")
    plt.xlabel(name)
    plt.ylabel("count")
    # only use full numbers as yticks. we want 4 ticks at most
    yticks = np.linspace(0, plt.yticks()[0][-1], 4)
    yticks = np.round(yticks).astype(int)  # Ensure full numbers
    plt.yticks(yticks)
    plt.tight_layout()
    
    
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def asses_reservoir(dir, plot_dir=None):
    print("assesing reservoir ", dir)
    if plot_dir is None:
        plot_dir = os.path.join(dir, "histplots")
    
    rhos = [get_rho(dir, i) for i in range(1, max_file_num(dir)+1)]
    stds = [get_rho_unc(dir, i) for i in range(1, max_file_num(dir)+1)]
    c1s = [get_c1(dir, i, rhos[i-1]) for i in range(1, max_file_num(dir)+1)]
    
    mean_rhos = [np.mean(rho) for rho in rhos]
    max_rhos = [np.max(rho) for rho in rhos]
    min_rhos = [np.min(rho) for rho in rhos]

    mean_c1s = [np.mean(c1) for c1 in c1s]
    mean_stds = [np.mean(std) for std in stds]
    mean_c1_stds = []

    for i in range(len(rhos)):
        rho = rhos[i]
        std = stds[i]
        if np.any(rho < 0) or np.any(std < 0):
            print(f"Negative values in rho or std in {i}")
            print(f"rho: {rho}")
            print(f"std: {std}")
        mean_c1_stds.append(np.mean(std[rho>0] / rho[rho>0]))


    mean_c1_stds = np.array(mean_c1_stds, dtype=float)
    Vext = []
    for i in range(len(rhos)):
        V = np.log(rhos[i]) + c1s[i]
        V[rhos[i] == 0] = np.nan
        Vext.append(V)
    mean_Vext = [np.mean(V[np.isfinite(V)]) for V in Vext]
    min_Vext = [np.min(V[np.isfinite(V)]) for V in Vext]
    max_Vext = [np.max(V[np.isfinite(V)]) for V in Vext]

    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    plot_histograms(mean_rhos, "Mean Density", plot_dir + "/mean_rho.png")
    plot_histograms(max_rhos, "Maximal Density", plot_dir + "/max_rho.png")
    plot_histograms(mean_c1s, r"Mean $c_1$", plot_dir + "/mean_c1.png")
    plot_histograms(mean_stds, r"Mean $\sigma_{\rho}$", plot_dir + "/mean_std.png")
    plot_histograms(mean_c1_stds, r"Mean $\sigma_{c_1}$", plot_dir + "/mean_c1_std.png")
    plot_histograms(min_Vext, r"Mean $\min{V_{\mathrm{ext}}}$", plot_dir + "/min_Vext.png")

    np.savetxt(os.path.join(plot_dir, "mean_rho.csv"), mean_rhos, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "max_rho.csv"), max_rhos, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "min_rho.csv"), min_rhos, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "mean_c1.csv"), mean_c1s, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "mean_std.csv"), mean_stds, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "mean_c1_stds.csv"), mean_c1_stds, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "mean_Vext.csv"), mean_Vext, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "min_Vext.csv"), min_Vext, delimiter=',')
    np.savetxt(os.path.join(plot_dir, "max_Vext.csv"), max_Vext, delimiter=',')

    print(f"Systems: {len(mean_rhos)}")
    print(f"mean rho: {np.mean(mean_rhos)} \tstd: {np.std(mean_rhos)}")
    print(f"mean max rho: {np.mean(max_rhos)} \tstd: {np.std(max_rhos)}")
    print(f"max max rho: {np.max(max_rhos)}")
    print(f"max mean rho: {np.max(mean_rhos)}")
    print(f"mean c1: {np.mean(mean_c1s)} \tstd: {np.std(mean_c1s)}")
    print(f"mean std: {np.mean(mean_stds)} \tstd: {np.std(mean_stds)}")
    print(f"mean c1 std: {np.mean(mean_c1_stds)} \tstd: {np.std(mean_c1_stds)}")
    print(f"mean min Vext: {np.mean(min_Vext)} \tstd: {np.std(min_Vext)}")


# function that renames the rho_i potential_i and rhostd_i files to a new random j index
def rename_files(datafolder:str):
    check_df_health(datafolder)
    max_file = max_file_num(datafolder)
    for i in range(1, max_file+1):
        os.rename(datafolder + f"/rho_{i}.csv", datafolder + f"/rho_{i}.csv.bak")
        os.rename(datafolder + f"/potential_{i}.json", datafolder + f"/potential_{i}.json.bak")
        os.rename(datafolder + f"/rhostd_{i}.csv", datafolder + f"/rhostd_{i}.csv.bak")
    
    file_numbers = np.random.permutation(max_file) + 1
    for i, j in enumerate(file_numbers):
        os.rename(datafolder + f"/rho_{i+1}.csv.bak", datafolder + f"/rho_{j}.csv")
        os.rename(datafolder + f"/potential_{i+1}.json.bak", datafolder + f"/potential_{j}.json")
        os.rename(datafolder + f"/rhostd_{i+1}.csv.bak", datafolder + f"/rhostd_{j}.csv")
    
    check_df_health(datafolder)
    print(f"Renamed {max_file} files in {datafolder} to random indices.")
    

def plot_data_folder(datafolder:str, plotfolder:str = None, L=10):
    
    plot_folder = os.path.join(datafolder, "plots")
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    
    num_files = max_file_num(datafolder)
    
    print("Plotting folder ", datafolder)
    for n in range(1, num_files + 1):
        print("Plotting ", n, "/", num_files, end="\r") # Print progress
        rho = get_rho(datafolder, n)
        # if rho.shape[0] > 100:
        #     downscale_factor = int(np.floor(rho.shape[0] / 100))
        #     rho = rho[::downscale_factor, ::downscale_factor]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Plot rho with aspect ration 1
        plt.colorbar(axes[0].imshow(rho, extent=[0, 10, 0, 10], aspect='equal', origin='lower'))
        axes[0].set_title(f"rho {n}")
        
        Vext = get_Vext(datafolder, n)
        
        # Plot V-mu
        plt.colorbar(axes[1].imshow(Vext, extent=[0, L, 0, L], aspect='equal', origin='lower'))
        axes[1].set_title(f"V-mu {n}")
        
        # Plot c1
        # ignore log(0) warnings
        c1 = get_c1(datafolder, n, rho)
        plt.colorbar(axes[2].imshow(c1, extent=[0, L, 0, L], aspect='equal', origin='lower'))
        axes[2].set_title(f"c1 {n}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"plot_{n}.png"))
        plt.close(fig)