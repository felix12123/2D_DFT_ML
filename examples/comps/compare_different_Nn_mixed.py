import src.training as trn
import src.training.models as _models
import os

Nn = [1, 2, 3, 4, 6, 8, 10, 12, 16, 32, 64]
epochs = 350

# inds = [i//2 if i % 2 == 1 else -i//2 for i in range(1,len(Nn))]
# Nn = [Nn[-i] for i in inds]
# print(Nn)

for n in Nn:
    mlt_new = trn.MLTraining(model=_models.Model_FMT2(20, n), L=10, dx=0.05, datafolder="/share/train_data/dx005-mixed-2", windowed=True, batchsize=64, num_workers=2, traintest_split=1, lr_start=1e-3, tensorboard_folder="runs/Nn_comp_mixed", batch_lr_decay=1-1e-5, min_lr=2e-6)
    
    # change folder to saved_trainings/Nncomp
    mlt_new.model_savepath = mlt_new.model_savepath.replace("saved_trainings", "saved_trainings/Nn_comp_mixed")
    
    if os.path.exists(mlt_new.model_savepath):
        mlt = trn.load_MLTraining(mlt_new.model_savepath)
    else:
        mlt = mlt_new
        
    # if mlt.trained_epochs >= epochs:
        # print(f"Model with Nn={n} already trained for {epochs} epochs, skipping.")
        # continue
    # print(f"Training model with Nn={n} for {epochs - mlt.trained_epochs} epochs.")
    
    if epochs > mlt.trained_epochs:
        mlt.train(epochs=epochs - mlt.trained_epochs)

