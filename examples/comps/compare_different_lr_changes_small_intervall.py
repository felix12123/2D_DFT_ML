import src.training as trn
import src.training.models as _models
import os
import numpy as np

import tensorboard.backend.event_processing.event_accumulator
Nn = 12
lr = 3e-5
lr_decs = 10 ** np.linspace(np.log10(5e-3), np.log10(2e-2), 8)

epochs = 500


for lr_c in lr_decs:
    mlt = trn.MLTraining(model=_models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=64, num_workers=2, traintest_split=0.8, lr_start=lr, lr_decay=1-lr_c, tensorboard_folder="runs/lr_dec_comp3", min_lr=0)
    
    # adjust savepath to include lr
    mlt.model_savepath = mlt.model_savepath.replace("saved_trainings", "saved_trainings/lr_dec_comp")
    lrc_str = str(mlt.lr_decay).replace(".", "_")
    mlt.model_savepath = mlt.model_savepath.split("_lr_dec")[0] + "_lr_decay" + lrc_str + ".pt"
    
    
    # if model has already been trained, load it
    if os.path.exists(mlt.model_savepath):
        mlt = trn.load_MLTraining(mlt.model_savepath)
    
    
    # train the model for the remaining epochs
    if epochs - mlt.trained_epochs <= 0: continue
    mlt.train(epochs=epochs - mlt.trained_epochs)
    
