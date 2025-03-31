import os
from src.training import *
import src.training.models as _models

Nn = 6
lr_start = 3e-5
lr_decay = 1 - 5e-3
mlt1 = MLTraining(model=_models.Model_FMT2_spherical(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", num_workers=2, traintest_split=0.8, lr_decay=lr_decay, lr_start=lr_start)

if os.path.exists(mlt1.model_savepath):
    mlt2 = load_MLTraining(mlt1.model_savepath)


mlt2.train(epochs=50)
# mlt1.show_c1_prediction()
