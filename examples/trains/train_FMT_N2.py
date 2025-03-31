import os
from src.training import *
import src.training.models as _models

Nn = 2
mlt1 = MLTraining(model=_models.Model_FMT(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin-lite", lr_start=3e-4)

# if os.path.exists(mlt1.savepath):
    # mlt1 = load_MLTraining(mlt1.savepath)


mlt1.train(epochs=100)

# mlt2 = MLTraining(model=models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=64, num_workers=2, traintest_split=0.8, lr_start=3e-3)

# if os.path.exists(mlt2.savepath):
#     mlt2 = load_MLTraining(mlt2.savepath)


# mlt2.train(epochs=100)
