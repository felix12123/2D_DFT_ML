import os
import re
import matplotlib.pyplot as plt
import numpy as np

from .training import load_MLTraining
from .training._plotting import plot_mean_loss_vs_Nn
from .training._bench import bench_iter, bench_iter_time, bench_c1
from .training._simfolder_utils import *


def compare_mlts(mlts, max_iter=500, tol=1e-6, model_names=None, savepath=None, time_itersteps=500, time_iterreps=5, title="", xlabel="", ylabel="", labels=None, comp_time=False):
    """Compares the performance of different models for the same potentials.
    """
    if isinstance(mlts, str):
        mlts = os.listdir(mlts)
        mlts = filter(os.path.isdir, mlts)
    # load mlts if they are paths
    if isinstance(mlts[0], str):
        mlts = [load_MLTraining(mlt) for mlt in mlts]

    if len(mlts) < 2:
        raise ValueError(
            "At least two MLTraining objects are required for comparison.")
        
    if model_names is None:
        model_names = [str(i+1) for i in range(len(mlts))]
    assert len(mlts) == len(model_names)


    # benchmark the models
    iter_losses = []
    iter_metrics = []
    c1_losses = []
    c1_metrics = []
    iter_times = []
    for i in range(len(mlts)):
        # load rho and c1 profiles
        rho_profiles = []
        c1_profiles = []
        for j in range(1, max_file_num(mlts[i].datafolder)+1):
            rho_profiles.append(get_rho(mlts[i].datafolder, j))
            c1_profiles.append(get_c1(mlts[i].datafolder, j, rho_profiles[-1]))
        all_c1_losses, all_c1_metrics = bench_c1(
            mlts[i], rhos=rho_profiles, c1s=c1_profiles)
        c1_losses.append(all_c1_losses.mean())
        c1_metrics.append(all_c1_metrics.mean())
        iter_loss, iter_metric, _ = bench_iter(mlts[i], max_iter=max_iter)
        iter_losses.append(iter_loss.mean())
        iter_metrics.append(iter_metric.mean())
        if comp_time:
            iter_times.append(bench_iter_time(
                mlts[i], repetitions=time_iterreps, iters=time_itersteps))

    # Plot the results with sci axis
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle("Comparison of model performance")

    for ax, data, title in zip(axes.flatten(),
                               [iter_losses, iter_metrics, c1_losses, c1_metrics],
                               ["Iterated loss", "Iterated metric", "C1 loss", "C1 metric"]):
        ax.bar(model_names, data)
        ax.set_title(title)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()

    if comp_time:
        fig_t = plt.figure(figsize=(4, 3), facecolor=(1, 1, 1, 0))
        fig_t.suptitle("Comparison of model speed")
        plt.bar(model_names, iter_times)
        plt.title("Time per iteration (s)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    if savepath is not None:
        print("plotted to ", savepath)
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)

        if comp_time:
            savepath_t = savepath.split(".")[0] + "_time.pdf"
            fig_t.savefig(savepath_t, bbox_inches='tight')
            plt.close(fig_t)

    else:
        return fig, fig_t


def compare_mlt_speed(mlts, iters=300, model_names=None, savepath=None, reps=5, xlabel=""):
    """Compares the speed of different models for the same potentials.
    """
    # load mlts if they are paths
    if isinstance(mlts, str):
        mlts = os.listdir(mlts)
        mlts = filter(lambda x: x.endswith(".pt"), mlts)
    if isinstance(mlts[0], str):
        mlts = [load_MLTraining(mlt) for mlt in mlts]

    if model_names is None:
        model_names = [str(i+1) for i in range(len(mlts))]
    assert len(mlts) == len(model_names)

    if len(mlts) < 2:
        raise ValueError(
            "At least two MLTraining objects are required for comparison.")

    iter_times = []
    iter_times_unc = []

    for i in range(len(mlts)):
        t, std = bench_iter_time(
            mlts[i], repetitions=reps, iters=iters, return_std=True)
        iter_times.append(t)
        iter_times_unc.append(std)

    # Plot the results with sci axis
    fig = plt.figure(figsize=(4, 3), facecolor=(1, 1, 1, 0))
    fig.suptitle("Comparison of model speed")
    if isinstance(model_names[0], str):
        plt.bar(model_names, iter_times)
        # add error bars
        plt.errorbar(model_names, iter_times, yerr=iter_times_unc,
                     fmt='none', ecolor='black', capsize=5)
        plt.xticks(rotation=45, ha='right')
    else:
        plt.scatter(model_names, iter_times, marker='x')
        # add error bars
        plt.errorbar(model_names, iter_times, yerr=iter_times_unc,
                     fmt='none', ecolor='black', capsize=5)
        plt.xlabel(xlabel)
        # fit a line
        p = np.polyfit(model_names, iter_times, 1)
        print("Time for Nn = 0: ", p[1])
        print("Time per Nn: ", p[0])
        print("For Nn = ", p[1]/p[0],
              " the convolutions take as much time as phi")
    plt.ylabel("Time per iteration (s)")
    plt.tight_layout()

    if savepath is not None:
        print("plotted to ", savepath)
        fig.savefig(savepath, bbox_inches='tight')

        # close plots
        plt.close(fig)

    else:
        return fig


def plot_Nn_comp(mltdir, plotfolder, exclude_Nn=[], name="", colorful=False):
    os.makedirs(plotfolder, exist_ok=True)

    def filename_to_Nn(filename):
        return int(re.search(r"(\d+)\.pt", filename).group(1))
    mlt_filenames = os.listdir(mltdir)
    mlt_filenames = sorted(mlt_filenames, key=filename_to_Nn)
    mlt_filenames = [
        p for p in mlt_filenames if filename_to_Nn(p) not in exclude_Nn]
    mlts = [load_MLTraining(os.path.join(mltdir, p)) for p in mlt_filenames]
    Ns = [filename_to_Nn(p) for p in mlt_filenames]

    c1_losses = []
    for mlt in mlts:
        losses, metrics, mean_rhos = bench_c1(mlt, max_iter=300)
        c1_losses.append(losses)
    yscale = "linear"
    if max(c1_losses) / min(c1_losses) > 10:
        yscale = "log"
    plot_mean_loss_vs_Nn(c1_losses, Ns, "", xscale="linear", figsize=(5, 4), xlabel=r"$k_{\mathrm{max}}$", yscale=yscale, ylabel=r"Mean Loss of $c_{1,\,\mathrm{ML}}$", highlight_Nn=8, colorful=colorful).savefig(
        f"{plotfolder}/Nn_comp_c1_loss{name}.pdf", bbox_inches='tight')

    iterlosses = []
    itermetrics = []
    itermean_rhos = []
    for mlt in mlts:
        losses, metrics, mean_rhos = bench_iter(mlt, max_iter=300)
        iterlosses.append(losses)
        itermetrics.append(metrics)
        itermean_rhos.append(mean_rhos)
    mean_losses = [np.mean([x for x in l if np.isfinite(x)])
                   for l in iterlosses]
    mean_metrics = [np.mean([x for x in m if np.isfinite(x)])
                    for m in itermetrics]

    plot_mean_loss_vs_Nn(mean_losses, Ns, "", figsize=(5, 4), xlabel=r"$k_{\mathrm{max}}$", ylabel=r"Mean Loss of $\rho_{\mathrm{ML}}$",
                         yscale=yscale, highlight_Nn=8, colorful=colorful).savefig(f"{plotfolder}/Nn_comp_iter_loss{name}.pdf", bbox_inches='tight')
    plot_mean_loss_vs_Nn(mean_metrics, Ns, "", figsize=(5, 4), xlabel=r"$k_{\mathrm{max}}$", ylabel=r"Mean Metric of $\rho_{\mathrm{ML}}$",
                         yscale=yscale, highlight_Nn=8, colorful=colorful).savefig(f"{plotfolder}/Nn_comp_iter_metric{name}.pdf", bbox_inches='tight')
