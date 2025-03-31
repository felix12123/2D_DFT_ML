from ._iter_folders import create_model_iteration_profiles
from ._simfolder_utils import max_file_num, get_Vext, get_rho, get_c1, get_rho_unc
from ._iter_folders import fmt_profiles_exist, create_FMT_profiles
from ._plotting import plot_rho_comp_small, plot_rho_comp_large, plot_metric_comps, create_loss_vs_mean_dens_plot, plot_one_metric, histogram_one_metric, plot_prediction
from ._training import tensor_to_np, np_to_tensor
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from typing import TYPE_CHECKING
if TYPE_CHECKING: # pragma: no cover
    from . import MLTraining


def check_simulation_accuracy(folder: str, L: int = 10):
    """Check the simulation accuracy by comparing the rho and c1 profiles to the FMT profiles."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found")
    print(f"Checking simulation accuracy in folder '{folder}'")

    os.makedirs(folder + "/DFT_comp", exist_ok=True)

    if not fmt_profiles_exist(folder):
        print("FMT profiles do not exist. Creating them...")
        create_FMT_profiles(folder)

    print("reading data...")
    num_profiles = max_file_num(folder)

    mean_rhos = []
    max_rhos = []
    mean_c1s = []
    mean_Vexts = []
    min_Vexts = []
    c1_losses = []
    c1_metrics = []
    c1_std = []
    rho_losses = []
    rho_metrics = []
    rho_std = []

    t0 = time.time()
    print("starting comparison...")
    for i in range(1, num_profiles+1):
        print("\rPlotting ", i, "/", num_profiles, end="")
        if i > 1:
            print("...   (time left: %.2f min)" % ((num_profiles-i+1) * (time.time() - t0)/60/(i-0.999)), end="")
        rho = get_rho(folder, i)
        rhostd = get_rho_unc(folder, i)
        c1 = get_c1(folder, i, rho)

        if np.mean(c1) > 0:
            ValueError("c1 should be negative")
        if np.mean(rho) < 0:
            ValueError("rho should be positive")

        Vext = np.log(rho) - c1
        # set Vext to inf where rho is zero
        Vext[rho <= 0] = np.inf
        if Vext.min() < -11:
            print(f"Skipped. Vext min: {Vext.min()}")
            continue

        rho_min = get_rho(folder + "/FMT", i)
        c1_n = get_c1(folder, i, rho_min)

        plot_rho_comp_small(Vext, rho, rho_min, r"$\rho_{\mathrm{sim}}$",
                            r"$\rho_{\mathrm{FMT}}$", folder + f"/DFT_comp/demo{i}.png", L=L)

        mean_rhos.append(np.mean(rho))
        max_rhos.append(np.max(rho))
        mean_c1s.append(np.mean(c1))
        mean_Vexts.append(np.mean(Vext))
        min_Vexts.append(np.min(Vext))
        c1_losses.append(np.mean((c1 - c1_n) ** 2))
        c1_metrics.append(np.mean(np.abs(c1 - c1_n)))
        rho_losses.append(np.mean((rho - rho_min) ** 2))
        rho_metrics.append(np.mean(np.abs(rho - rho_min)))
        rho_std.append(rhostd)
        c1_std.append(rhostd / rho)
        c1_std[-1][rho <= 0] = 0

    print("\rfinished!\t\t\t\t")

    plot_one_metric(max_rhos, rho_metrics, folder + "/DFT_comp/report_rho_metric_vs_max_rho.png", "linear", "linear",
                    "max rho", "rho metric", "Relation between max rho and rho metric")

    histogram_one_metric(rho_metrics, folder + "/DFT_comp/report_rho_metric_hist.png",
                         "rho metric", "count",
                         "Histogram of rho metric\nmean rho metric = %.2e" % np.mean(rho_metrics))
    histogram_one_metric(rho_losses, folder + "/DFT_comp/report_rho_loss_hist.png",
                         "rho loss", "count",
                         "Histogram of rho loss\nmean rho loss = %.2e" % np.mean(rho_losses))
    histogram_one_metric(c1_losses, folder + "/DFT_comp/report_c1_loss_hist.png",
                         "c1 loss", "count",
                         "Histogram of c1 loss\nmean c1 loss = %.2e" % np.mean(c1_losses))
    histogram_one_metric(c1_metrics, folder + "/DFT_comp/report_c1_metric_hist.png",
                         "c1 metric", "count",
                         "Histogram of c1 metric\nmean c1 metric = %.2e" % np.mean(c1_metrics))

    plot_one_metric(mean_rhos, rho_metrics, folder + "/DFT_comp/report_rho_metric_vs_mean_rho.png", "linear", "log",
                    "mean rho", "rho metric", "Relation between mean rho and rho metric\nmean rho metric = %.2e" %
                    np.mean(rho_metrics))

    plot_one_metric(mean_rhos, np.array(rho_metrics) / np.array(mean_rhos),
                    folder + "/DFT_comp/report_rel_rho_metric_vs_mean_rho.png", "linear", "log",
                    "mean rho", "relative rho metric", "Mean rho vs. relative rho difference\nmean relative difference = %.2e" %
                    np.mean(np.array(rho_metrics) / np.array(mean_rhos)))

    plot_one_metric(mean_Vexts, np.array(rho_metrics) / np.array(mean_rhos),
                    folder + "/DFT_comp/report_rel_rho_metric_vs_mean_Vext.png", "linear", "log",
                    "mean Vext", "relative rho metric", "Mean Vext vs. relative rho difference\nmean relative difference = %.2e" %
                    np.mean(np.array(rho_metrics) / np.array(mean_rhos)))

    plot_one_metric(max_rhos, c1_losses, folder + "/DFT_comp/report_c1_loss_vs_max_rho.png", "linear", "log",
                    "max rho", "c1 loss", "Relation between max rho and c1 loss\nmean c1 loss = %.2e" %
                    np.mean(c1_losses))

    msg = ""
    msg += f"mean rho = {np.mean(mean_rhos)}\n"
    msg += f"max metric = {np.max(rho_metrics)}\n"
    msg += f"mean c1 = {np.mean(mean_c1s)}\n"
    msg += f"mean Vext = {np.mean(mean_Vexts)}\n"
    msg += f"min min Vext = {np.min(min_Vexts)}\n"
    msg += f"mean min Vext = {np.mean(min_Vexts)}\n"
    msg += f"mean c1 loss = {np.mean(c1_losses)}\n"
    msg += f"mean c1 metric = {np.mean(c1_metrics)}\n"
    msg += f"mean rho loss = {np.mean(rho_losses)}\n"
    msg += f"mean rho metric = {np.mean(rho_metrics)}\n"
    msg += f"mean rho std from simulation = {np.mean(rho_std)}\n"

    np.savetxt(folder + "/DFT_comp/report.txt", [msg], fmt="%s")

def get_default_mliter_folder(self:'MLTraining', potfolder):
    """Get the default model iteration folder."""
    model_iter_folder = self.workspace + "/ML_iter/" + potfolder.split("/")[-1]
    return model_iter_folder
    
def demo_model_iteration(self:'MLTraining', plotfolder=None, potfolder=None, compfolder=None, model_iter_folder=None, max_iter: int = 1000, eps: float = 1e-5, comparison_vs="Sim", L=10, Vext_integration_n=7):
    """
    Demonstrates the model iteration process by generating and comparing density profiles 
    from machine learning predictions and reference data. This function also creates plots 
    and metrics to evaluate the performance of the model.
    Args:
        plotfolder (str, optional): Path to the folder where plots will be saved. If None, 
            a default folder is created based on the comparison type and workspace.
        potfolder (str, optional): Path to the folder containing potential profiles. If None, 
            defaults to the data folder of the MLTraining instance. Expects potential_i.json
            files in potfolder/pot.
        compfolder (str, optional): Path to the folder containing comparison profiles. If None, 
            it is determined based on the `comparison_vs` argument. Expects rho_i.csv files
            in compfolder/rho.
        model_iter_folder (str, optional): Path to the folder where model iteration profiles 
            are stored. If None, a default folder is created. Expects rho_i.csv files in
            model_iter_folder/rho.
        max_iter (int, optional): Maximum number of iterations for the model to converge. 
            Defaults to 1000.
        eps (float, optional): Convergence tolerance for the model. Defaults to 1e-5.
        comparison_vs (str, optional): Specifies the type of comparison. Can be "Sim" for 
            simulation or "FMT" for Fundamental Measure Theory. Defaults to "Sim".
        L (int, optional): System size parameter for plotting and calculations. Defaults to 10.
        Vext_integration_n (int, optional): Number of integration points for external potential. 
            Defaults to 7.
    Notes:
        - If FMT profiles do not exist in the specified folder, they are created automatically.
        - If model iteration profiles are missing or incomplete, they are generated.
        - Metrics and plots are saved to the specified or default `plotfolder`.
    Example:
        mlt.demo_model_iteration(
            plotfolder="plots/model_iterations",
            potfolder="data/simulations",
            compfolder="data/simulations",
            model_iter_folder="mlt/ML_iter/simulations",
            max_iter=500,
            eps=1e-6,
            comparison_vs="Sim",
            L=10,
            Vext_integration_n=7
        )
    """
    
    if compfolder == None and potfolder == None:
        if comparison_vs == "Sim":
            compfolder = self.datafolder
        elif comparison_vs == "FMT":
            compfolder = self.datafolder + "/FMT"
        else:
            raise ValueError("comparison_vs should be 'Sim' or 'FMT'")

    if compfolder == None:
        if comparison_vs == "Sim":
            compfolder = potfolder
        elif comparison_vs == "FMT":
            compfolder = potfolder + "/FMT"
    if potfolder == None:
        potfolder = self.datafolder
    if model_iter_folder is None:
        model_iter_folder = get_default_mliter_folder(self, potfolder)
    if plotfolder is None:
        if comparison_vs == "Sim":
            plotfolder = self.workspace + "/plots/iter_vs_sim/" + potfolder.split("/")[-1]
        elif comparison_vs == "FMT":
            plotfolder = model_iter_folder + "/plots/iter_vs_fmt/" + potfolder.split("/")[-1]
        else:
            raise ValueError("comparison_vs should be 'Sim' or 'FMT'")
    os.makedirs(plotfolder, exist_ok=True)
    if not os.path.exists(potfolder):
        raise FileNotFoundError(f"Folder '{potfolder}' not found")

    print("Getting Potentials from ", potfolder)
    print("Getting comparison profiles from ", compfolder)
    print("Getting model iteration profiles from ", model_iter_folder)
    print("Plotting to ", plotfolder)

    num_profiles = max_file_num(potfolder)
    if comparison_vs == "FMT":
        if not fmt_profiles_exist(potfolder):
            print("FMT profiles do not exist. Creating them...")
            create_FMT_profiles(potfolder, savefolder=compfolder,
                                eps=eps, L=L, Vext_integration_n=Vext_integration_n)

    if not os.path.exists(model_iter_folder) or max_file_num(model_iter_folder) < num_profiles:
        if os.path.exists(model_iter_folder):
            print("ML profiles do not exist for all profiles. There are only ", max_file_num(
                model_iter_folder), " profiles and we need ", num_profiles)
        else:
            print("ML profiles do not exist. Creating them...")
        create_model_iteration_profiles(self, potfolder=potfolder, ml_iter_folder=model_iter_folder,
                                        tol=eps, max_iter=max_iter, L=L, Vext_integration_n=Vext_integration_n)

    self.model = self.model.to(self.device)
    self.model.eval()

    mean_rhos = []
    max_rhos = []
    mean_Vexts = []
    min_Vexts = []
    rho_losses = []
    rho_metrics = []
    max_diff_rho = []

    print("Plotting to ", plotfolder)
    for i in range(1, 1+num_profiles):
        print("Plotting ", i, "/", num_profiles, end="\r")

        rho_comp = get_rho(compfolder, i)
        Vext = get_Vext(potfolder, i, Vext_integration_n)
        rho_ml = get_rho(model_iter_folder, i)

        plot_rho_comp_small(Vext, rho_ml, rho_comp, r"$\rho_{\mathrm{ML}}$",
                            r"$\rho_{\mathrm{" + comparison_vs + "}}$", plotfolder + f"/comp_{i}.pdf", L=L)

        mean_rhos.append(np.mean(rho_comp[rho_comp > 0]))
        max_rhos.append(np.max(rho_comp[rho_comp > 0]))
        mean_Vexts.append(np.mean(Vext[rho_comp > 0]))
        min_Vexts.append(np.min(Vext[rho_comp > 0]))
        rho_losses.append(np.mean((rho_comp - rho_ml)[rho_comp > 0] ** 2))
        rho_metrics.append(
            np.mean(np.abs(rho_comp[rho_comp > 0] - rho_ml[rho_comp > 0])))
        max_diff_rho.append(np.max(np.abs(rho_comp - rho_ml)))
    print()

    plot_metric_comps(mean_rhos, max_rhos, None, mean_Vexts, min_Vexts, None,
                      None, None, rho_losses, rho_metrics, None, plotfolder + "/metrics")
    fig = create_loss_vs_mean_dens_plot(simfolder=potfolder, ml_iter_folder=model_iter_folder,
                                        train_test_split=self.traintest_split, comp_folder=compfolder, comp_index="Sim")
    fig.savefig(plotfolder + "/rho_metric_vs_mean_rho.pdf", bbox_inches='tight')

    # create a csv file that saves the metrics for each profile
    metrics = np.array([list(range(1, 1+num_profiles)), mean_rhos, max_rhos,
                       mean_Vexts, min_Vexts, max_diff_rho, rho_losses, rho_metrics]).T
    # add column names
    metrics = np.vstack((["Profile", "Mean rho", "Max rho", "Mean Vext",
                        "Min Vext", "Max diff rho", "rho loss", "rho metric"], metrics))
    np.savetxt(plotfolder + "/metrics.csv", metrics, delimiter=",", fmt="%s")
    msg = "5 worst systems:\n"
    for i in np.argsort(rho_metrics)[-5:]:
        msg += "Profile %d: rho metric = %.2e\n" % (i+1, rho_metrics[i])
    msg += "\n5 best systems:\n"
    for i in np.argsort(rho_metrics)[:5]:
        msg += "Profile %d: rho metric = %.2e\n" % (i+1, rho_metrics[i])
    np.savetxt(plotfolder + "/worst_best.txt", [msg], fmt="%s")


def demo_model_iteration2(self, plotfolder=None, simfolder=None, fmt_folder=None, model_iter_folder=None, max_iter: int = 1000, eps: float = 1e-5, L=10, Vext_integration_n=7):
    """Similar to demo_model_iteration, but compares the model iteration profiles to the simulation profiles and FMT profiles."""
    if simfolder == None:
        simfolder = self.datafolder
    if fmt_folder is None:
        fmt_folder = simfolder + "/FMT"
    if model_iter_folder is None:
        model_iter_folder = get_default_mliter_folder(self, simfolder)
    if plotfolder is None:
        plotfolder = self.workspace + "/plots/iter_vs_sim_fmt/" + simfolder.split("/")[-1]
    os.makedirs(plotfolder, exist_ok=True)
    if not os.path.exists(simfolder):
        raise FileNotFoundError(f"Folder '{simfolder}' not found")

    print("Getting Potentials from ", simfolder)
    print("Getting model iteration profiles from ", model_iter_folder)
    print("getting FMT profiles from ", fmt_folder)
    print("getting sim profiles from ", simfolder)
    print("Plotting to ", plotfolder)

    num_profiles = max_file_num(simfolder)
    if not fmt_profiles_exist(simfolder, fmt_folder=fmt_folder):
        print("FMT profiles do not exist. Creating them...")
        create_FMT_profiles(simfolder, savefolder=fmt_folder,
                            eps=eps, L=L, Vext_integration_n=Vext_integration_n)

    if not os.path.exists(model_iter_folder) or max_file_num(model_iter_folder) < num_profiles:
        print("ML profiles do not exist. Creating them...")
        create_model_iteration_profiles(self, potfolder=simfolder, ml_iter_folder=model_iter_folder,
                                        tol=eps, max_iter=max_iter, Vext_integration_n=Vext_integration_n)

    self.model = self.model.to(self.device)
    self.model.eval()

    for i in range(1, 1+num_profiles):
        print("Plotting ", i, "/", num_profiles, end="\r")
        rho_fmt = get_rho(fmt_folder, i)
        rho_sim = get_rho(simfolder, i)

        Vext = get_Vext(simfolder, i)

        rho_ml = get_rho(model_iter_folder, i)

        plot_rho_comp_large(Vext, rho_ml, rho_sim, rho_fmt,
                            plotfolder + f"/comp_{i}.pdf", L=self.L)

    fig = create_loss_vs_mean_dens_plot(simfolder=simfolder, ml_iter_folder=model_iter_folder,
                                        train_test_split=self.traintest_split, comp_folder=simfolder, comp_index="Sim")
    fig.savefig(plotfolder + "/rho_metric_vs_mean_rho.pdf")



def show_c1_prediction(self):
    """
    Generates and saves c1 predictions using the trained model.

    This method evaluates the model on data in the specified folder, generates 
    c1 predictions, and saves the plots in a designated folder.

    Args:
        - None
    Note:
        - Saves plots in "workspace/plots/c1_prediction/<datafolder_name>".
        - Uses helper functions like `max_file_num`, `get_rho`, `get_c1`, 
          `np_to_tensor`, `tensor_to_np`, and `plot_prediction`.
    """
    if not os.path.exists(self.datafolder):
        raise FileNotFoundError(f"Folder '{self.datafolder}' not found")

    self.model = self.model.to(self.device)
    self.model.eval()
    # add padding
    self.model.add_padding()
    df = self.datafolder

    plot_folder = self.workspace + "/plots/c1_prediction/" + df.split("/")[-1]
    os.makedirs(plot_folder, exist_ok=True)
    print(f"Saving c1 predictions in folder '{plot_folder}'")

    with torch.no_grad():
        for i in range(1, max_file_num(df)+1):
            rho = get_rho(df, i)
            c1 = get_c1(df, i, rho)

            inputs, labels = np_to_tensor(rho).to(self.device), np_to_tensor(c1).to(self.device)
            outputs = self.model(inputs)

            c1_neural = tensor_to_np(outputs)
            c1_neural[rho <= 0] = np.nan
            plot_prediction(rho, c1, c1_neural, plot_folder + f"/c1_{i}.png")
            