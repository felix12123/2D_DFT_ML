import src.training as trn
import src.training.models as _models
import os


Nn = 8
ms = [_models.Model_FMT, _models.Model_FMT2, _models.Model_FMT_one_conv, _models.Model_FMT_small]
epochs = 15


for model in ms:
    mlt_new = trn.MLTraining(model=model(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-1e11s-sin", windowed=True, batchsize=256, num_workers=2, traintest_split=0.8, lr_start=3e-3, batch_lr_decay=1-2e-4, tensorboard_folder="runs/Model_comparison")
    
    mlt_new.model_savepath = mlt_new.model_savepath.replace("saved_trainings", "saved_trainings/Modelcomp")
    
    if os.path.exists(mlt_new.model_savepath):
        mlt = trn.load_MLTraining(mlt_new.model_savepath)
    else:
        mlt = mlt_new
    if mlt.trained_epochs >= epochs:
        print(f"Model {model} already trained for {epochs} epochs, skipping.")
        continue
    print(f"Training model {model} for {epochs - mlt.trained_epochs} epochs.")
    mlt.train(epochs=epochs - mlt.trained_epochs)

