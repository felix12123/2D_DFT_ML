import src.training as trn
import src.training.models as models
import os
from math import log2

Nn = 12
lr_start = 3e-5
lr_decay = 1 - 5e-3
epochs = 250
bss = [8, 16, 32, 64, 128, 256, 512]
bss.sort(key=lambda x: abs(log2(x)-log2(64)))
print(bss)

for bs in bss:
    print("\n\nstarting training with bs: ", bs)
    lr_s = lr_start * bs / 64
    mlt_new = trn.MLTraining(model=models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=bs, num_workers=2, traintest_split=1, lr_start=lr_s, tensorboard_folder="runs/bs_comparison_lr_norm", lr_decay=lr_decay, min_lr=0)
    
    mlt_new.model_savepath = mlt_new.model_savepath.split(".pt")[0] + f"_bs{bs}.pt"
    mlt_new.model_savepath = mlt_new.model_savepath.replace("saved_trainings", "saved_trainings/bscomp_lr_norm")
    
    print("savepath: ", mlt_new.model_savepath)
    if os.path.exists(mlt_new.model_savepath):
        mlt = trn.load_MLTraining(mlt_new.model_savepath)
    else:
        mlt = mlt_new
    
    mlt.train(epochs=epochs - mlt.trained_epochs)

    