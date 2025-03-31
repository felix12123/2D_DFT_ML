import src.training as trn
from src.training.models import Model_FMT2
from src.training.models import SphericalConv2d
import os
from config import SIN_TRAIN_PARAMS as tparams

Nn = 8
lrs = tparams["lrs"]
lrd = tparams["lrd"]
bs = tparams["bs"]
df = "/share/train_data/dx005-1e11s-sin"
epochs = 1000



### Normal Training ###
mlt = trn.MLTraining(model=Model_FMT2(20, Nn), L=10, dx=0.05, datafolder=df, lr_start=lrs, lr_decay=lrd, batchsize=bs, min_lr=0)
mlt.model_savepath = mlt.model_savepath.replace(".pt", "_final.pt")

# if model has already been trained, load it
if os.path.exists(mlt.model_savepath):
    mlt = trn.load_MLTraining(mlt.model_savepath)
    print(mlt.savepath)


# train the model for the remaining epochs
mlt.train(epochs=epochs - mlt.trained_epochs)
    

### Spherical Training ###

epochs = 1000
mlt_sphere = trn.MLTraining(model=Model_FMT2(20, Nn, convtype=SphericalConv2d), L=10, dx=0.05, datafolder=df, lr_start=lrs, lr_decay=lrd, batchsize=bs, min_lr=0)
mlt_sphere.model_savepath = mlt_sphere.model_savepath.replace(".pt", "_sphere_final.pt")

# if model has already been trained, load it
if os.path.exists(mlt_sphere.model_savepath):
    mlt_sphere = trn.load_MLTraining(mlt_sphere.model_savepath)
    
    
# train the model for the remaining epochs
print("Training spherical model")
print("Trained epochs: ", mlt_sphere.trained_epochs)
print("Remaining epochs: ", epochs - mlt_sphere.trained_epochs)
mlt_sphere.train(epochs=epochs - mlt_sphere.trained_epochs)
print("Training spherical model done")


### Spherical Training with Smoothing ###

mlt_sphere_smooth = trn.MLTraining(model=Model_FMT2(20, Nn, convtype=SphericalConv2d), L=10, dx=0.05, datafolder=df, lr_start=lrs, lr_decay=lrd, batchsize=bs, min_lr=0)
mlt_sphere_smooth.model_savepath = mlt_sphere_smooth.model_savepath.replace(".pt", "_sphere_smooth_final.pt")

# if model has already been trained, load it
if os.path.exists(mlt_sphere_smooth.model_savepath):
    mlt_sphere_smooth = trn.load_MLTraining(mlt_sphere_smooth.model_savepath)

# train the model for the remaining epochs
epochs = [100] + 6 * [50] + [100] + 6 * [50] + [100]
smooths = [0.5] * (len(epochs)-1)
mlt_sphere_smooth.train_smooth(epochs=epochs, smooths=smooths, device="cuda")
