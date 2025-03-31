import os
from src.training import *
import src.training.models as models

Nn = 12
lr_start = 3e-5
lr_decay = 1 - 5e-3
epochs = 250
bs = 64
datafolder = "/share/train_data/dx005-1e11s-sin"
model = models.Model_FMT2

mlt1 = MLTraining(model=model(20, Nn), L=10, dx=0.05, datafolder=datafolder, windowed=True, batchsize=bs, num_workers=2, traintest_split=1, lr_start=lr_start, lr_decay=lr_decay, min_lr=0, shuffle=True, tensorboard_folder="runs/shuffle_comp")
mlt1.model_savepath = mlt1.model_savepath.replace(".pt", "_shuffle.pt").replace("saved_trainings", "saved_trainings/shuffle_comp")
mlt2 = MLTraining(model=model(20, Nn), L=10, dx=0.05, datafolder=datafolder, windowed=True, batchsize=bs, num_workers=2, traintest_split=1, lr_start=lr_start, lr_decay=lr_decay, min_lr=0, shuffle=False, tensorboard_folder="runs/shuffle_comp")
mlt2.model_savepath = mlt2.model_savepath.replace(".pt", "_noshuffle.pt").replace("saved_trainings", "saved_trainings/shuffle_comp")

if os.path.exists(mlt1.model_savepath):
    mlt1 = load_MLTraining(mlt1.model_savepath)
if os.path.exists(mlt2.model_savepath):
    mlt2 = load_MLTraining(mlt2.model_savepath)

mlt1.train(epochs=300 - mlt1.trained_epochs)
mlt2.train(epochs=300 - mlt2.trained_epochs)
