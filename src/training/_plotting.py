import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import re



from src.training._pot import load_potential_params
from src.training._simfolder_utils import *
from src.training import *

def float_to_sci_string(f, digits=2):
    s = "{:.{digits}e}".format(f, digits=digits)
    base, exponent = s.split("e")
    exponent = int(exponent)
    base = base.rstrip('0').rstrip('.')
    return str(base) + "\\cdot 10^{" + str(exponent) + "}"

def plot_rho_comp_small(Vext, rho1, rho2, rho1_name, rho2_name, savepath, L=10):
    """ Plots a comparison of the densities from the ML model and the simulation."""
    rho_diff = rho1 - rho2
    # rho_diff[np.logical_or(rho1==0, rho2==0)] = 0
    rho1n = rho1_name.replace("$", "")
    rho2n = rho2_name.replace("$", "")
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    
    # imshow Vext, rhom, rhos, rho_diff in one plot and add colorbars
    plt.rc('font', size=12)
    fig, ax = plt.subplots(2, 2, figsize=(7, 6)) # create figure with 2x2 subplots
    
    # set titles with size 16
    plt.rc('font', size=16)
    ax[1, 1].set_title(r"$" + rho1n + "-" + rho2n + "$")
    # ax[1, 1].set_title(r"$\Delta \rho=" + rho1n + "-" + rho2n + r"\\\langle| \Delta \rho| \rangle = " + float_to_sci_string(abs(rho_diff).mean()) + "$")  
    ax[0, 0].set_title(r"$\beta(V_{\mathrm{ext}}-\mu)$")
    ax[0, 1].set_title(rho1_name)
    ax[1, 0].set_title(rho2_name)
    
    # add labels a) b) c) d) below the subplots
    ax[0, 0].set_title("a)", fontfamily='serif', loc='left', fontsize=14)
    ax[0, 1].set_title("b)", fontfamily='serif', loc='left', fontsize=14)
    ax[1, 0].set_title("c)", fontfamily='serif', loc='left', fontsize=14)
    ax[1, 1].set_title("d)", fontfamily='serif', loc='left', fontsize=14)
    

    # plot Vext, rhom, rhos, rho_diff with smaller font size
    plt.rc('font', size=12)

    # Vext
    im0 = ax[0, 0].imshow(Vext)
    ax[0, 0].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[0, 0].set_xticklabels([0, int(L/2), int(L)])
    ax[0, 0].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[0, 0].set_yticklabels([0, int(L/2), int(L)])
    ax[0, 0].set_ylabel(r"$y/\sigma$")
    fig.colorbar(im0, ax=ax[0, 0])

    # rhom
    im1 = ax[0, 1].imshow(rho1)
    ax[0, 1].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[0, 1].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im1, ax=ax[0, 1])
    ax[0, 1].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[0, 1].set_yticklabels([0, int(L/2), int(L)])
    
    
    # rhos
    im2 = ax[1, 0].imshow(rho2)
    ax[1, 0].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[1, 0].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im2, ax=ax[1, 0])
    ax[1, 0].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[1, 0].set_yticklabels([0, int(L/2), int(L)])
    ax[1, 0].set_ylabel(r"$y/\sigma$")
    ax[1, 0].set_xlabel(r"$x/\sigma$")
    
    # rho_diff
    max_diff = max(abs(rho_diff.min()), abs(rho_diff.max()))
    if max_diff == 0:
        max_diff = 1
    norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff)
    im3 = ax[1, 1].imshow(rho_diff, cmap='coolwarm', norm=norm)
    ax[1, 1].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[1, 1].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im3, ax=ax[1, 1])
    ax[1, 1].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[1, 1].set_yticklabels([0, int(L/2), int(L)])
    ax[1, 1].set_xlabel(r"$x/\sigma$")
    
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

def plot_rho_comp_large(Vext, rhoML, rhoSIM, rhoFMT, savepath, L=10):
    """ Plots a comparison of the densities from the ML model, the simulation and the FMT model."""
    rho_diffSIM = rhoML - rhoSIM
    rho_diffFMT = rhoML - rhoFMT
    # rho_diffSIMn = "$\\rho_{\\mathrm{ML}} - \\rho_{\\mathrm{SIM}}$"
    # rho_diffFMTn = "$\\rho_{\\mathrm{ML}} - \\rho_{\\mathrm{FMT}}$"
    
    
    rhoFMTname = "$\\rho_{\\mathrm{FMT}}$"
    rhoSIMname = "$\\rho_{\\mathrm{SIM}}$"
    rhoMLname = "$\\rho_{\\mathrm{ML}}$"
    rhoFMTn = rhoFMTname.replace("$", "")
    rhoSIMn = rhoSIMname.replace("$", "")
    rhoMLn = rhoMLname.replace("$", "")
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    
    # imshow Vext, rhom, rhos, rho_diff in one plot and add colorbars
    plt.rc('font', size=12)
    fig, ax = plt.subplots(3, 2, figsize=(7, 9)) # create figure with 2x2 subplots
    
    
    Vext_pos = [0, 0]
    rhoML_pos = [0, 1]
    rhoSIM_pos = [1, 0]
    rhodiffSIM_pos = [1, 1]
    rhoFMT_pos = [2, 0]
    rhodiffFMT_pos = [2, 1]
    
    # set titles with size 16
    plt.rc('font', size=16)
    ax[*Vext_pos].set_title(r"$\beta (V_{\mathrm{ext}}-\mu)$")
    ax[*rhoML_pos].set_title(rhoMLname)
    ax[*rhoSIM_pos].set_title(rhoSIMname)
    ax[*rhoFMT_pos].set_title(rhoFMTname)
    ax[*rhodiffSIM_pos].set_title(r"$" + rhoMLn + "-" + rhoSIMn + "$")# + r"\\\langle| \Delta \rho| \rangle = " + float_to_sci_string(abs(rho_diffSIM).mean()) + "$")
    ax[*rhodiffFMT_pos].set_title(r"$" + rhoMLn + "-" + rhoFMTn + "$")# + r"\\\langle| \Delta \rho| \rangle = " + float_to_sci_string(abs(rho_diffFMT).mean()) + "$")
    
    # set labels a) b) c) d) e) f)
    ax[*Vext_pos].set_title("a)", fontfamily='serif', loc='left')
    ax[*rhoML_pos].set_title("b)", fontfamily='serif', loc='left')
    ax[*rhoSIM_pos].set_title("c)", fontfamily='serif', loc='left')
    ax[*rhodiffSIM_pos].set_title("d)", fontfamily='serif', loc='left')
    ax[*rhoFMT_pos].set_title("e)", fontfamily='serif', loc='left')
    ax[*rhodiffFMT_pos].set_title("f)", fontfamily='serif', loc='left')
    
    
    # set axis labels
    ax[*Vext_pos].set_ylabel(r"$y/\sigma$")
    ax[*rhoSIM_pos].set_ylabel(r"$y/\sigma$")
    ax[*rhoFMT_pos].set_ylabel(r"$y/\sigma$")
    ax[*rhoFMT_pos].set_xlabel(r"$x/\sigma$")
    ax[*rhodiffFMT_pos].set_xlabel(r"$x/\sigma$")
    
    

    # plot Vext, rhom, rhos, rho_diff with smaller font size
    plt.rc('font', size=12)

    # Vext
    im0 = ax[*Vext_pos].imshow(Vext)
    ax[*Vext_pos].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[*Vext_pos].set_xticklabels([0, int(L/2), int(L)])
    ax[*Vext_pos].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[*Vext_pos].set_yticklabels([0, int(L/2), int(L)])
    fig.colorbar(im0, ax=ax[*Vext_pos])

    # rhom
    im1 = ax[*rhoML_pos].imshow(rhoML)
    ax[*rhoML_pos].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhoML_pos].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im1, ax=ax[*rhoML_pos])
    ax[*rhoML_pos].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhoML_pos].set_yticklabels([0, int(L/2), int(L)])
    
    # rhos
    im2 = ax[*rhoSIM_pos].imshow(rhoSIM)
    ax[*rhoSIM_pos].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhoSIM_pos].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im2, ax=ax[*rhoSIM_pos])
    ax[*rhoSIM_pos].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhoSIM_pos].set_yticklabels([0, int(L/2), int(L)])
    
    # rhofmt
    im2 = ax[*rhoFMT_pos].imshow(rhoFMT)
    ax[*rhoFMT_pos].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhoFMT_pos].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im2, ax=ax[*rhoFMT_pos])
    ax[*rhoFMT_pos].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhoFMT_pos].set_yticklabels([0, int(L/2), int(L)])
    
    
    # rho_diff SIM
    max_diff = max(abs(rho_diffSIM.min()), abs(rho_diffSIM.max()))
    norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff)
    im3 = ax[*rhodiffSIM_pos].imshow(rho_diffSIM, cmap='coolwarm', norm=norm)
    ax[*rhodiffSIM_pos].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhodiffSIM_pos].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im3, ax=ax[*rhodiffSIM_pos])
    ax[*rhodiffSIM_pos].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhodiffSIM_pos].set_yticklabels([0, int(L/2), int(L)])
    
    # rho_diff FMT
    max_diff = max(abs(rho_diffFMT.min()), abs(rho_diffFMT.max()))
    norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff)
    im4 = ax[*rhodiffFMT_pos].imshow(rho_diffFMT, cmap='coolwarm', norm=norm)
    ax[*rhodiffFMT_pos].set_xticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhodiffFMT_pos].set_xticklabels([0, int(L/2), int(L)])
    fig.colorbar(im4, ax=ax[*rhodiffFMT_pos])
    ax[*rhodiffFMT_pos].set_yticks([0, len(Vext)//2, len(Vext)-1])
    ax[*rhodiffFMT_pos].set_yticklabels([0, int(L/2), int(L)])
    
    
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def get_quantity(sim_folder, iter_folder, quantities: list, pot_quantities: list = [], compfolder=None):
    """ Get quantities from the simulation and the ML model.
    Args:
        sim_folder (str): Path to the simulation folder.
        iter_folder (str): Path to the ML model folder.
        quantities (list): List of functions that take rho_sim and rho_min as input and return a quantity.
        pot_quantities (list): List of functions that take a potential file as input and return a quantity.
    Returns:
        data (list): List of quantities for each profile.
        pot_data (list): List of quantities for each potential file.
    """
    if not os.path.exists(sim_folder):
        raise ValueError(f"Simulation folder {sim_folder} does not exist.")
    if compfolder is None:
        compfolder = sim_folder
    
    num_profiles = max_file_num(sim_folder)
    if num_profiles == 0:
        raise ValueError(f"Found no profiles in {iter_folder}.")
        
    data = [np.zeros(num_profiles) for _ in quantities]
    pot_data = [[None for _ in range(num_profiles)] for _ in pot_quantities]
    
    # preallocate data
    rho_comp = get_rho(compfolder, 1)
    rho_min = get_rho(iter_folder, 1)
    for i in range(1, 1+num_profiles):
        np.copyto(rho_comp, get_rho(compfolder, i))
        np.copyto(rho_min, get_rho(iter_folder, i))
        for j in range(len(quantities)):
            data[j][i-1] = quantities[j](rho_sim=rho_comp, rho_min=rho_min)
        for j in range(len(pot_quantities)):
            pot_data[j][i-1] = pot_quantities[j](sim_folder + f"/pot/potential_{i}.json")
    return data, pot_data


def which_potential(potfile):
        """ Determine the type of potential from the potential file.
        Args:
            potfile (str): Path to the potential file.
        Returns:
            str: Type of potential. Can be "box", "plat" or "sin".
        """
        pot_params = load_potential_params(potfile)
        if pot_params["wall"] == "b":
            return "box"
        elif pot_params["wall"] == "n" and len(pot_params["plat_heights"]) > 0:
            return "plat"
        else:
            return "sin"

def create_loss_vs_mean_dens_plot(simfolder, ml_iter_folder, train_test_split, comp_index = "Sim", comp_folder=None):
    """ Create a plot of the mean density vs the loss for different potentials. Training and test data are
    distinguished by different markers.
    Args:
        simfolder (str): Path to the simulation folder.
        ml_iter_folder (str): Path to the ML model folder.
        train_test_split (float): Fraction of training data.
        comp_index (str): Index of the component to compare. Default is "Sim".
        comp_folder (str): Path to the folder to compare with. Default is None, which means the same as simfolder.
    Returns:
        fig: Figure object.
    """
    if comp_folder is None:
        comp_folder = simfolder
    # define metric functions
    rho_metric = lambda rho_sim, rho_min: np.mean(np.abs(rho_sim[rho_sim>0] - rho_min[rho_sim>0]))
    mean_rho =   lambda rho_sim, rho_min: np.mean(rho_sim[rho_sim>0])
    quantities = [rho_metric, mean_rho]

    # get metric data
    data, pot_data = get_quantity(simfolder, ml_iter_folder, quantities, [which_potential], compfolder=comp_folder)
    rho_metrics = data[0]
    mean_rhos = data[1]
    pot_types = np.atleast_1d(pot_data[0])
    
    # get indices for different potentials
    box_indices = np.where(pot_types == 'box')[0]
    plat_indices = np.where(pot_types == 'plat')[0]
    sin_indices = np.where(pot_types == 'sin')[0]
    indice_groups = [box_indices, plat_indices, sin_indices]
    indice_names = ["Hard Walls + Plateaus", "Plateau", r"$V_{\mathrm{sin}}$ only"]

    # plot rho_metrics vs mean_rhos
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    
    N_train = int(len(rho_metrics) * train_test_split)
    
    # scatter rho_metrics vs mean_rhos for different potentials. Use different markers for train and test data
    colors = plt.cm.tab10
    for i in range(3):
        color = colors(2-i)
        for index in indice_groups[i]:
            marker_shape = "o" if index < N_train else "^"
            label = None if index != indice_groups[i][0] else indice_names[i]
            ax.scatter(mean_rhos[index], rho_metrics[index], marker=marker_shape, label=label, color=color, facecolors='none')
    
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\langle \rho \rangle$")
    ax.set_ylabel(r"$\langle |\rho_{\mathrm{" + comp_index + r"}} - \rho_{\mathrm{ML}}| \rangle$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.title(r"$\langle |\rho_{\mathrm{" + comp_index + r"}} - \rho_{\mathrm{ML}}| \rangle$ vs $\langle \rho_{\mathrm{" + comp_index + r"}} \rangle$")
    plt.tight_layout()
    if sum([ig.shape[0] > 0 for ig in indice_groups]) == 1:
        return fig
    plt.legend()
    return fig


def plot_one_metric(xmetric, ymetric, savepath, yscale="linear", xscale="linear", title="", xlabel="", ylabel="", figsize=(6, 4)):
    """ Plot one metric vs another metric and save the plot."""
    plt.figure(figsize=figsize, facecolor=(1,1,1,0))
    plt.scatter(xmetric, ymetric)
    if max(np.abs(np.array(xmetric))) < 1e-2:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    if max(np.abs(np.array(ymetric))) < 1e-2:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.ylim(get_ylims(ymetric, yscale))
    plt.yscale(yscale)
    plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

def histogram_one_metric(metric, savepath, title="", xlabel="", ylabel="", bins=50, figsize=(6, 4)):
    """ Plot a histogram of one metric and save the plot."""
    plt.figure(figsize=figsize, facecolor=(1,1,1,0))
    plt.hist(metric, bins=bins)
    plt.title(title)
    if max(np.abs(np.array(metric))) < 1e-2:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
def plot_metric_comps(mean_rhos=None, max_rhos=None, mean_c1s=None, mean_Vexts=None, min_Vexts=None, c1_losses=None, c1_uncert=None, c1_metrics=None, rho_losses=None, rho_metrics=None, times=None, plotfolder="media"):
    """ Plot metric comparisons for different parameters."""
    os.makedirs(plotfolder, exist_ok=True)
    if mean_rhos is not None:
        comp_metrics = [c1_losses, c1_metrics, rho_losses, rho_metrics]
        titles = [r"$c_1$ MSE vs $\langle \rho \rangle$", r"$c_1$ MAE vs $\langle \rho \rangle$", r"$\rho$ MSE vs $\langle \rho \rangle$", r"$\rho$ MAE vs $\langle \rho \rangle$"]
        xlabels = [r"$\langle \rho \rangle$", r"$\langle \rho \rangle$", r"$\langle \rho \rangle$", r"$\langle \rho \rangle$"]
        ylabels = [r"$c_1$ MSE", r"$c_1$ MAE", r"$\rho$ MSE", r"$\rho$ MAE"]
        filename = ["c1_mse_vs_rho.pdf", "c1_mae_vs_rho.pdf", "rho_mse_vs_rho.pdf", "rho_mae_vs_rho.pdf"]
        for i in range(4):
            if comp_metrics[i] is not None:
                plot_one_metric(mean_rhos, comp_metrics[i], plotfolder + "/" + filename[i], title=titles[i], xlabel=xlabels[i], ylabel=ylabels[i], yscale="log")
    if max_rhos is not None:
        comp_metrics = [c1_losses, c1_metrics, rho_losses, rho_metrics]
        titles = [r"$c_1$ MSE vs $\langle \rho \rangle$", r"$c_1$ MAE vs $\langle \rho \rangle$", r"$\rho$ MSE vs $\langle \rho \rangle$", r"$\rho$ MAE vs $\langle \rho \rangle$"]
        xlabels = [r"$\langle \rho \rangle$", r"$\langle \rho \rangle$", r"$\langle \rho \rangle$", r"$\langle \rho \rangle$"]
        ylabels = [r"$c_1$ MSE", r"$c_1$ MAE", r"$\rho$ MSE", r"$\rho$ MAE"]
        filename = ["c1_mse_vs_maxrho.pdf", "c1_mae_vs_maxrho.pdf", "rho_mse_vs_maxrho.pdf", "rho_mae_vs_maxrho.pdf"]
        for i in range(4):
            if comp_metrics[i] is not None:
                plot_one_metric(max_rhos, comp_metrics[i], plotfolder + "/" + filename[i], title=titles[i], xlabel=xlabels[i], ylabel=ylabels[i], yscale="log")
    if min_Vexts is not None:
        comp_metrics = [c1_losses, c1_metrics, rho_losses, rho_metrics]
        titles = [r"$c_1$ MSE vs $\langle V_{\mathrm{ext}}-\mu \rangle$", r"$c_1$ MAE vs $\langle V_{\mathrm{ext}}-\mu \rangle$", r"$\rho$ MSE vs $\langle V_{\mathrm{ext}}-\mu \rangle$", r"$\rho$ MAE vs $\langle V_{\mathrm{ext}}-\mu \rangle$"]
        xlabels = [r"$\langle V_{\mathrm{ext}}-\mu \rangle$", r"$\langle V_{\mathrm{ext}}-\mu \rangle$", r"$\langle V_{\mathrm{ext}}-\mu \rangle$", r"$\langle V_{\mathrm{ext}}-\mu \rangle$"]
        ylabels = [r"$c_1$ MSE", r"$c_1$ MAE", r"$\rho$ MSE", r"$\rho$ MAE"]
        filename = ["c1_mse_vs_minVext.pdf", "c1_mae_vs_minVext.pdf", "rho_mse_vs_minVext.pdf", "rho_mae_vs_minVext.pdf"]
        for i in range(4):
            if comp_metrics[i] is not None:
                plot_one_metric(min_Vexts, comp_metrics[i], plotfolder + "/" + filename[i], title=titles[i], xlabel=xlabels[i], ylabel=ylabels[i], yscale="log")
    
    if c1_uncert is not None and c1_metrics is not None:
        # plot c1 uncert vs c1 metric and shade area green where c1 metric is below c1 uncert
        plt.figure(figsize=(6, 4), facecolor=(1,1,1,0))
        plt.scatter(c1_uncert, c1_metrics)
        xlim = plt.xlim
        ylim = plt.ylim
        plt.fill_between(c1_uncert, c1_metrics, c1_uncert, where=c1_metrics < c1_uncert, color='green', alpha=0.3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(r"$c_1$ metric vs $c_1$ uncertainty")
        plt.xlabel(r"$\sigma_{c_1}$")
        plt.ylabel(r"$\langle | c_{1, \mathrm{sim}} - c_{1, \mathrm{min}} | \rangle$")
        plt.savefig(plotfolder + "/c1_uncert_vs_c1_metric.pdf", bbox_inches='tight')
        plt.close()

def plot_g_r(distances, densities, comparison, comp_name, labels, save_path=None):
    """ Plot g(r) for different densities and comparison.
    Args:
        distances (list): List of distances for each density.
        densities (list): List of densities for each distance.
        comparison (list): List of comparison densities for each distance.
        comp_name (str): Name of the comparison density.
        labels (list): List of labels for each density.
        save_path (str): Path to save the plot. If None, the plot is shown instead.
    """
    assert len(distances) == len(densities) == len(labels) == len(comparison)
    
    plt.rc('font', size=16)
    fig = plt.figure(figsize=(5, 4), facecolor=(1,1,1,0))
    
    for i in range(len(densities)):
        # plot densities and comparison
        plt.plot(distances[i], densities[i], label=labels[i])
        plt.plot(distances[i], comparison[i], linestyle="--", alpha=0.5, linewidth=1, color="black")
    plt.plot([], [], label=comp_name, color="black", linestyle="--") # add legend entry for comparison
    
    plt.xlabel(r"$r/\sigma$")
    plt.xlim(0.0, 5.0)
    plt.ylim(bottom=0)
    plt.ylabel(r"$\rho_0g(r)$")
    
    plt.rc('font', size=12) # set font size to 12 for legend
    plt.legend()
    plt.rc('font', size=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        return fig



def get_ylims(values, yscale="linear"):
    """ Get ylims for the plot based on the values and yscale."""
    values = np.array(values).flatten()
    val_range = max(values) - min(values)
    if yscale=="linear":
        buffer = 0.1 * val_range
        return min(values) - buffer, max(values) + buffer
    elif yscale=="log":
        buffer = np.log10(max(values) / min(values)) * 0.1
        
        return 10 ** (np.log10(min(values)) - buffer), 10 ** (np.log10(max(values)) + buffer)
    else:
        raise ValueError("Invalid yscale. Choose 'linear' or 'log'.")


def plot_mean_loss_vs_Nn(losses, Nns, title="Mean Loss vs. Parameter", figsize=(5,4), yscale="linear", xscale="linear", xlabel="Parameter", ylabel="Mean Loss", highlight_Nn=None, colorful=True):
    """ Plot mean loss vs. Nn."""
    plt.rc('font', size=16)
    plot = plt.figure(dpi=300, figsize=figsize, facecolor=(1,1,1,0))
    if colorful:
        colors = plt.cm.plasma(np.linspace(0, 1, len(losses)+2))[:len(losses)] # exclude the last color to avoid to light yellows
    else:
        colors = [plt.cm.tab10(0)] * len(losses)
    
    if highlight_Nn is not None:
        highlight_index = Nns.index(highlight_Nn)
        plt.scatter(Nns[highlight_index], losses[highlight_index], marker='o', s=175, facecolors='none', edgecolors='red')
        
    for i in range(len(losses)):
        plt.errorbar(Nns[i], losses[i], fmt='x', capsize=2, capthick=1, elinewidth=1, markersize=7, c=colors[i])
    
    plt.yscale(yscale)
    plt.xscale(xscale)
    if yscale == "linear" and max(losses) < 1e-2:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.ylim(get_ylims(losses, yscale))
    
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.subplots_adjust(bottom=0.15)
    return plot


def plot_prediction(rho, c1, c1_neural, plot_path, cmap='viridis'):
    """
    Plot input, target, prediction, and absolute difference with colorbars.

    Args:
        rho (np.ndarray): The input tensor.
        c1 (np.ndarray): The ground truth tensor (target).
        c1_neural (np.ndarray): The predicted tensor.
        plot_path (str): Path to save the plot.
        cmap (str): Colormap to use for the plots.
    """
    # Determine the min and max values for the color scale
    vmin = min(c1.min().item(), c1_neural.min().item())
    vmax = max(c1.max().item(), c1_neural.max().item())

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot input
    im0 = axs[0, 0].imshow(rho, cmap=cmap)
    axs[0, 0].set_title('Input')
    fig.colorbar(im0, ax=axs[0, 0])

    # Plot target
    im1 = axs[0, 1].imshow(c1, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Target')
    fig.colorbar(im1, ax=axs[0, 1])

    # Plot prediction
    im2 = axs[1, 0].imshow(c1_neural, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Prediction')
    fig.colorbar(im2, ax=axs[1, 0])

    # Plot absolute difference
    abs_diff = np.abs(c1_neural - c1)
    im3 = axs[1, 1].imshow(abs_diff, cmap=cmap)
    axs[1, 1].set_title('Absolute Difference\n(mean = %.2e)' % abs_diff.mean().item())
    fig.colorbar(im3, ax=axs[1, 1])

    # Save the plot
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    
    
    
    
# This file contains utility functions for visualising the training results.
def create_shared_kernel_matrix(kernels, padding=4):
    """
    Create a large matrix that contains all kernels in a grid with some padding between the kernels.
    pad with nan.
    """
    num_kernels = len(kernels)
    kernel_size = kernels[0].shape[1]
    
    num_rows = int(np.ceil(np.sqrt(num_kernels)))
    num_cols = int(np.ceil(num_kernels / num_rows))
    
    kernel_matrix = np.full((num_rows * (kernel_size + padding) - padding, num_cols * (kernel_size + padding) - padding), np.nan)
    
    for i, kernel in enumerate(kernels):
        row = i // num_cols
        col = i % num_cols
        kernel_matrix[row * (kernel_size + padding):row * (kernel_size + padding) + kernel_size, col * (kernel_size + padding):col * (kernel_size + padding) + kernel_size] = kernel
        
    kernel_matrix = np.pad(kernel_matrix, ((padding, padding), (padding, padding)), mode='constant', constant_values=np.nan)
    
    return kernel_matrix

def collect_all_kernels(conv2d_layer:nn.Conv2d):
    """
    Collect all kernels from a Conv2d layer and return them as a list of 2d arrays.
    """
    kernels = []
    for i in range(conv2d_layer.weight.shape[0]):
        kernel = conv2d_layer.weight[i].detach().cpu().numpy().copy()
        kernels.append(kernel)
    return kernels


def show_kernels(self, output_file='', num_kernels=None, conv_layer=0, title=""):
    # Ensure the model is on the CPU
    # Get the first convolutional layer
    kernels = self.model.collect_all_kernels()[conv_layer]
    
    if num_kernels is None:
        num_kernels = len(kernels)
    
    # Create a shared kernel matrix
    kernel_matrix = create_shared_kernel_matrix(kernels[:num_kernels])
    
    # Plot the kernel matrix
    plt.rc("font", size=10)
    fig = plt.figure(figsize=(6, 6), facecolor=(1,1,1,0))
    max_diff = max(abs(kernel_matrix[np.isfinite(kernel_matrix)].min()), abs(kernel_matrix[np.isfinite(kernel_matrix)].max()))
    norm = TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff)
    plt.imshow(kernel_matrix, cmap='coolwarm', norm=norm)
    plt.colorbar()
    plt.xticks([]) # remove ticks
    plt.yticks([])
    plt.box(False) # remove box
    if title != "":
        plt.rc("font", size=16)
        plt.title(title)
        plt.rc("font", size=10)
    plt.tight_layout()
    
    
    # Save the plot to a file
    if output_file != "":
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    else:
        return fig