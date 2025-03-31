import src.training as trn
from src.training.models import Model_FMT2
import os
from config import SIN_TRAIN_PARAMS as params


Nn = [1, 2, 3, 4, 8, 12, 16, 32, 64]
epochs = 350
lrs = params["lrs"]
lrd = params["lrd"]
bs = params["bs"]




for n in Nn:
    mlt_new = trn.MLTraining(model=Model_FMT2(20, n), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=bs, num_workers=2, traintest_split=0.8, lr_start=lrs, tensorboard_folder="runs/Nn_comp", lr_decay=lrd, min_lr=0)
    
    # change folder to saved_trainings/Nncomp
    mlt_new.model_savepath = mlt_new.model_savepath.replace("saved_trainings", "saved_trainings/Nn_comp")
    
    if os.path.exists(mlt_new.model_savepath):
        mlt = trn.load_MLTraining(mlt_new.model_savepath)
        print("Loaded existing training")
        print("Tensorboard folder: ", mlt.tensorboard_folder)
        if mlt.tensorboard_folder != mlt_new.tensorboard_folder:
            print("Updating tensorboard folder to ", mlt_new.tensorboard_folder)
            print()
        mlt.tensorboard_folder = mlt_new.tensorboard_folder
        mlt.save()
        continue
    else:
        mlt = mlt_new
    
    if epochs > mlt.trained_epochs:
        mlt.train(epochs=epochs - mlt.trained_epochs)

