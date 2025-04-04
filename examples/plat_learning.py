# This File contains a simple example of using the MLTraining class on
# a sin and plateau potential based simulation folder
# To run this example, you need to import this file from the main.py file

# Importing the necessary libraries
# Add the src directory to the Python path
import os
import numpy as np
import shutil


from src.training import MLTraining, load_MLTraining
from src.training.models import Model_Bay, Model_FMT, Model_FMT2
from src.create_df import create_td
from src.reservoir_utils import plot_data_folder
from config import PLAT_TRAIN_PARAMS as params


# create a simulation folder with FMT
dx = 0.05
L = 10
num_systems = 50
datafolder = "examples/data/plat-dx{dx:.2f}".format(dx=dx).replace(".", "_")


# create a folder with random potentials if it does not exist
# or if it does not contain the required number of systems
if (
    not os.path.exists(datafolder) or
    not os.path.exists(datafolder + "/rho/rho_" + str(num_systems) + ".csv")
):
    create_td(
        datafolder,     # path to the folder
        num_systems,    # number of systems
        1e-6,           # tolerance for rho
        L,              # length of the box
        dx,             # grid spacing
        8,             # max number of sinusoids
        0.2,            # min and max sin amplitude
        0.8,
        2,              # max period
        4,              # max number of plateaus
        1,              # max height of plateaus
        0.75,           # min and max size of plateaus
        3.0,
        0,              # min and max rotation of plateaus
        np.pi/2,
        np.pi/6,        # min and max angle of parallelograms
        np.pi/2,
        ["n", "b"],     # wall types ("n" for no wall, "b" for box)
        0.1,            # min and max wall height
        1.0,            # min and max wall height
        -1.0,           # min and max mu
        5.0)
# plot profiles of the data folder
if not os.path.exists(datafolder + "/plots"):
    plot_data_folder(datafolder)


# create a model
model = Model_FMT2(int(np.ceil(1/dx)), Nn=16,
                   hidden_channels=48, hidden_layers=4)

# create a MLTraining object
mlt = MLTraining("examples/plat_mlt_dx" + ("%.2f" % dx).replace(".", "_"), model=model, datafolder=datafolder, L=L, dx=dx,
                 batchsize=params["bs"], lr_start=params["lrs"], lr_decay=params["lrd"], windowed=True)

# load the model if it exists
if os.path.exists(mlt.model_savepath):
    mlt = load_MLTraining(mlt.workspace)
    print("loaded training session from %s" % mlt)


# train the model
mlt.train(70 - mlt.trained_epochs)


# demo the model
if os.path.exists(mlt.workspace + "/ML_iter"):
    shutil.rmtree(mlt.workspace + "/ML_iter")
mlt.show_kernels()
mlt.show_c1_prediction()
mlt.demo_model_iteration()
mlt.g_r()
