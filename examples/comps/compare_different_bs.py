import src.training as trn
import src.training.models as models
import os

Nn = 12
lr_start = 3e-5
lr_decay = 1 - 5e-3
epochs = 175
bss = [8, 16, 32, 64, 128, 256, 512]

for bs in bss:
    print("\n\nstarting training with bs: ", bs)
    mlt_new = trn.MLTraining(model=models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=bs, num_workers=2, traintest_split=1, lr_start=lr_start, tensorboard_folder="runs/bs_comparison", lr_decay=lr_decay, min_lr=0)
    
    mlt_new.model_savepath = mlt_new.model_savepath.split(".pt")[0] + f"_bs{bs}.pt"
    mlt_new.model_savepath = mlt_new.model_savepath.replace("saved_trainings", "saved_trainings/bscomp")
    
    print("savepath: ", mlt_new.model_savepath)
    if os.path.exists(mlt_new.model_savepath):
        mlt = trn.load_MLTraining(mlt_new.model_savepath)
    else:
        mlt = mlt_new
    
    mlt.train(epochs=epochs - mlt.trained_epochs)

    