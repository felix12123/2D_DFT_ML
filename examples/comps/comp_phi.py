import os

from src.training import *
from src.training.models import Model_FMT2
from config import SIN_TRAIN_PARAMS as params

df = "/share/td/dx005-1e11s-sin"
L = 10
dx = 0.05
Nn = 8

lrd = params["lrd"]
lrs = params["lrs"]
bs = params["bs"]
epochs = 500

phi_layers = [3, 6, 9]
phi_nodes = [32, 64, 128]

# save all combinations of phi_layers and phi_nodes
phi_params = []
for pl in phi_layers:
    for pn in phi_nodes:
        phi_params.append((pl, pn))
# sort phi_params by the product of the two values
phi_params.sort(key=lambda x: x[0] * x[1])
print(phi_params)
# for each combination of phi_layers and phi_nodes, create a new MLTraining object and train it
for pl, pn in phi_params:
    workspace = "saved_trns/phi_comp/phi_%d_%d" % (pl, pn)
    os.makedirs(workspace, exist_ok=True)
    mlt = MLTraining(workspace, Model_FMT2(20, Nn, hidden_channels=pn, hidden_layers=pl), L=L, dx=dx, datafolder=df, windowed=True, batchsize=bs, num_workers=2, traintest_split=1, lr_start=lrs, lr_decay=lrd, min_lr=0)
    
    
    print("savepath: ", mlt.workspace)
    if os.path.exists(mlt.workspace) and os.path.exists(mlt.workspace + "/model.pt"):
        mlt = load_MLTraining(mlt.workspace)
            
    mlt.train(epochs=epochs - mlt.trained_epochs)
    
    
    



