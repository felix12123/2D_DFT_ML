import src.training as trn
from src.training.models import Model_FMT2
import os


Nn = 8
epochs = 350
lrs = 3e-5
lrd = 1-5e-3
bs = 64

num_runs = 11


for i in range(num_runs):
    mlt = trn.MLTraining(model=Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", num_workers=2, lr_start=lrs, lr_decay=lrd, batchsize=bs, tensorboard_folder="runs/variations", min_lr=0)
    
    mlt.model_savepath = mlt.model_savepath.replace("saved_trainings", "saved_trainings/variations")
    mlt.model_savepath = mlt.model_savepath.replace(".pt", f"_{i}.pt")
    
    # if model has already been trained, load it
    if os.path.exists(mlt.model_savepath):
        mlt = trn.load_MLTraining(mlt.model_savepath)
        
    # train the model for the remaining epochs
    mlt.train(epochs=epochs - mlt.trained_epochs)
    
