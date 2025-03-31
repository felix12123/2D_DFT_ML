import os
from src.training import *
import src.training.models as _models

Nn = 16
mlt1 = MLTraining(model=_models.Model_FMT2_spherical(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", num_workers=2, traintest_split=0.8)

# if os.path.exists(mlt1.savepath):
    # mlt1 = load_MLTraining(mlt1.savepath)


mlt1.train(epochs=250)
# mlt1.show_c1_prediction()
