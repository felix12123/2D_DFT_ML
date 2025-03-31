import shutil
import os
import pytest
import torch
from src.training import MLTraining, estimate_num_batches, load_MLTraining
from src.training.models import Model_Bay, Model_FMT, Model_FMT2

import torch.nn as nn

# Mock TOTAL_MEMORY for testing
TOTAL_MEMORY = 1e9

# Mock max_file_num for testing
@pytest.mark.order(0)
def test_estimate_num_batches(test_res):
    datafolder = "tests/test_data"
    batchsize = 64
    traintest_split = 0.8
    L = 10
    dx = 0.05
    win_rad_bins = 20

    num_batches = estimate_num_batches(datafolder, batchsize, traintest_split, L, dx, win_rad_bins)
    assert num_batches > 0

@pytest.mark.order(0)
def test_MLTraining_initialization(test_res):
    folder = "tests/mock_workspace_FMT"
    L = 10
    dx = 0.05
    model = Model_Bay(int(1/dx), 4)
    datafolder = "tests/test_data"
    batchsize = 2
    total_batches = 20

    training_instance = MLTraining(
        folder=folder,
        model=model,
        L=L,
        dx=dx,
        datafolder=datafolder,
        total_batches=total_batches,
        batchsize=batchsize,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        lr_start=1e-3,
        lr_decay=0.93,
        windowed=True
    )
    training_instance.save() # will be used later

    assert training_instance.workspace == folder
    assert training_instance.model == model
    assert training_instance.L == L
    assert training_instance.dx == dx
    assert training_instance.datafolder == datafolder
    assert training_instance.batchsize == batchsize
    assert training_instance.total_batches == total_batches
    assert training_instance.device in ["cuda", "cpu"]
    assert training_instance.lr_decay == 0.93
    assert training_instance.windowed == True
    
    training_instance = MLTraining(
        folder="tests/mock_workspace_FMT2",
        model=Model_FMT2(int(1/dx), 4),
        L=L,
        dx=dx,
        datafolder=datafolder,
        total_batches=total_batches,
        batchsize=batchsize,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        lr_start=1e-3,
        lr_decay=0.93,
        windowed=True
    )
    training_instance.save()
    assert training_instance.workspace == "tests/mock_workspace_FMT2"

@pytest.mark.order(0)
def test_MLTraining_invalid_datafolder():
    folder = "tests/mock_workspace"
    model = Model_Bay(20, 4)
    L = 10
    dx = 0.5
    datafolder = "invalid_datafolder"

    with pytest.raises(ValueError, match="Data folder does not exist"):
        MLTraining(folder=folder, model=model, L=L, dx=dx, datafolder=datafolder)

@pytest.mark.order(0)
def test_load_MLTraining(test_res):
    folder = "tests/mock_workspace_Bay"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = 10
    dx = 0.05
    datafolder = "tests/test_data"
    Nn = 4
    model = Model_FMT(int(1/dx), Nn)
    
    mlt = MLTraining(
        folder=folder,
        model=model,
        L=L,
        dx=dx,
        datafolder=datafolder,
        total_batches=20,
        batchsize=2,
        device=device,
        lr_start=1e-3,
        lr_decay=0.93,
        windowed=True
    )
    mlt.save()
    
    # test if the model is saved correctly
    instance = load_MLTraining(folder, device)
    

    assert instance.workspace == folder
    assert isinstance(instance.model, nn.Module)
    assert instance.L == L
    assert instance.dx == dx
    assert instance.datafolder == datafolder
    assert isinstance(instance.criterion, nn.MSELoss)
    assert isinstance(instance.optimizer, torch.optim.Adam)
    assert isinstance(instance.metric, nn.L1Loss)
    
    # test set_workspace
    new_folder = "tests/mock_workspace_Bay_nowindows"
    instance.set_workspace(new_folder)
    instance.windowed = False
    instance.save()
    assert instance.workspace == new_folder
    assert instance.model_savepath == new_folder + "/model.pt"
    assert instance.optimizer_savepath == new_folder + "/optimizer.pt"
    assert instance.scheduler_savepath == new_folder + "/scheduler.pt"
    assert instance.train_logs_savepath == new_folder + "/records/train_logs.csv"
    assert instance.test_logs_savepath == new_folder + "/records/test_logs.csv" 
    # check if the files are moved
    assert os.path.exists(new_folder + "/model.pt")
    assert os.path.exists(new_folder + "/optimizer.pt")
    assert os.path.exists(new_folder + "/scheduler.pt")
    assert os.path.exists(new_folder + "/params.json")
    