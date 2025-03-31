import os
from src.training import *
import src.training.models as _models

Nn = 12
mlt1 = MLTraining(model=_models.Model_FMT2(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-mixed-3", windowed=True, batchsize=64, num_workers=2, traintest_split=0.9, lr_start=1e-3, batch_lr_decay=1-1e-5, min_lr=2e-6)
    
# if os.path.exists(mlt1.savepath):
    # mlt1 = load_MLTraining(mlt1.savepath)



print("Loss before training  with new folder: ", mlt1.calc_loss())
mlt1.train(epochs=15)

print("Loss after training  with new folder: ", mlt1.calc_loss())

try:
    mlt1.g_r()
except:
    pass
try:
    mlt1.demo_model_performance()
except:
    pass

