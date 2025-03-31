import src.training as trn
import src.training.models as _models
import os


Nn = 12
ms = [_models.Model_FMT_one_conv, _models.Model_FMT2, _models.Model_FMT_small]
epochs = 150


for model in ms:
    mlt_new = trn.MLTraining(model=model(20, Nn), L=10, dx=0.05, datafolder="/share/train_data/dx005-mixed-3", windowed=True, batchsize=64, num_workers=2, traintest_split=1, lr_start=1e-3, tensorboard_folder="runs/Model_comp_mixed")
    mlt_new.lr_decay = mlt_new.lr_decay
    
    mlt_new.model_savepath = mlt_new.model_savepath.replace("saved_trainings", "saved_trainings/Model_comp_mixed")
    
    if os.path.exists(mlt_new.model_savepath):
        mlt = trn.load_MLTraining(mlt_new.model_savepath)
    else:
        mlt = mlt_new
    if mlt.trained_epochs >= epochs:
        print(f"Model {model} already trained for {epochs} epochs, skipping.")
        continue
    print(f"Training model {model} for {epochs - mlt.trained_epochs} epochs.")
    mlt.train(epochs=epochs - mlt.trained_epochs)


