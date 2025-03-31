# compare if smoothing the kernel helps with the training. 
# it might be interesting to compare the iteration quality of the kernels

import os
import src.training as trn
import src.training.models as models
from config import SIN_TRAIN_PARAMS as tparams

Nn = 8
lrd = tparams["lrd"]
lrs = tparams["lrs"]
bs = tparams["bs"]
epochs  = [75,  75, 20, 20, 20, 20, 20, 20, 20, 20]
smooths = [0.5, 1,  1,  1,  1,  1,  1,  1,  1,  1]
epochs = epochs + [tparams["epochs"] - sum(epochs)]

mlt = trn.MLTraining(model=models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", batchsize=bs, num_workers=2, traintest_split=0.8, lr_start=lrs, lr_decay=lrd)

# add _smooth to the savepath to indicate that the kernel was smoothed
mlt.model_savepath = mlt.model_savepath[:-3] + "_smooth.pt"

if os.path.exists(mlt.model_savepath):
    mlt = trn.load_MLTraining(mlt.model_savepath)


mlt.train_smooth(epochs, smooths)