import os
from src.training import *
import src.training.models as models

Nn = 8
mlt1 = MLTraining(model=models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=256, num_workers=2, traintest_split=0.8, lr_decay=0.995, lr_start=3e-3)

if os.path.exists(mlt1.model_savepath):
    mlt1 = load_MLTraining(mlt1.model_savepath)

print("loss after loading: ", mlt1.calc_loss())
mlt1.train(epochs=250)
# mlt1.show_c1_prediction()
