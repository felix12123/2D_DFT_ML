import os
from src.training import *
import src.training.models as _models
import numpy as np

Nn = 64
dx = 0.05
datafolder = "/share/train_data/dx005-1e12s-sin-ha"
mlt1 = MLTraining(model=_models.Model_FMT(int(np.round(1/dx)), Nn), L=10, dx=dx, datafolder=datafolder, windowed=True, batchsize=256, num_workers=2, traintest_split=0.8, save_interval=5, lr_decay=0.98, lr_start=1e-3, tensorboard_folder="runs_ha")
if os.path.exists(mlt1.model_savepath):
    mlt1 = load_MLTraining(mlt1.model_savepath)

mlt1.train(epochs=250)
