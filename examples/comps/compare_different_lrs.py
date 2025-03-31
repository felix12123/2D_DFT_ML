import src.training as trn
import src.training.models as _models
import os


Nn = 12
lrs = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]#, 1e-3, 3e-3, 1e-2, 3e-2]
epochs = 20


for lr in lrs:
    mlt = trn.MLTraining(model=_models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=64, num_workers=2, traintest_split=0.8, lr_start=lr,lr_decay=1, tensorboard_folder="runs/lr_comp")
    
    # adjust savepath to include lr
    mlt.model_savepath = mlt.model_savepath.replace("saved_trainings", "saved_trainings/lr_comp")
    mlt.model_savepath = mlt.model_savepath.replace(".pt", f"_lr" + str(lr).replace(".", "_") + ".pt")
    
    # if model has already been trained, load it
    if os.path.exists(mlt.model_savepath):
        mlt = trn.load_MLTraining(mlt.model_savepath)
    
    # train the model for the remaining epochs
    if epochs - mlt.trained_epochs <= 0: continue
    mlt.train(epochs=epochs - mlt.trained_epochs)
    
