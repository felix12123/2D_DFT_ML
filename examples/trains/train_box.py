import os
from src.training import *
import src.training.models as _models

Nn = 32
mlt1 = MLTraining(model=_models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-5e11s-box-2", windowed=True, batchsize=128, num_workers=2, traintest_split=0.8, lr_start=2e-3)

# mlt1.savepath = mlt1.savepath.split(".pt")[0] + f"_3.pt"
# if os.path.exists(mlt1.savepath):
    # mlt1 = load_MLTraining(mlt1.savepath)

mlt1.train(epochs=100)