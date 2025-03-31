import os
from src.training import *
import src.training.models as _models

Nn = 6
lr_start = 1e-4
lr_decay = 1 - 5e-3
mlt1 = MLTraining(model=_models.Model_FMT2_spherical(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-mixed-3", num_workers=2, traintest_split=0.8, lr_decay=lr_decay, lr_start=lr_start)
mlt1.model_savepath = mlt1.model_savepath.replace("saved_trainings", "saved_trainings/sphere")

if os.path.exists(mlt1.model_savepath):
    mlt1 = load_MLTraining(mlt1.model_savepath)

mlt1.demo_model_performance(device="cuda")

# mlt1.train(epochs=250)
# mlt1.show_c1_prediction()
