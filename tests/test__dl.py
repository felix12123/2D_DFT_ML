from src.training._dl import *
from src.training._simfolder_utils import max_file_num
import pytest
import numpy as np

@pytest.mark.order(1)
def test_get_train_window_dls(test_res):
    folder = "tests/test_data"
    win_rad = 5
    total_batches = 3
    batchsize = 32
    split = 0.7
    shuffle = True
    percent_used = None
    num_workers = 1

    train_dl, rho_profiles, c1_profiles = get_train_window_dls(folder, win_rad, total_batches, batchsize, split, shuffle, percent_used, num_workers)
    assert len(rho_profiles) == max_file_num(folder)
    assert len(c1_profiles) == max_file_num(folder)
    assert len(train_dl) == total_batches
    for batch in train_dl:
        rho_batch, c1_batch = batch
        assert rho_batch.shape == (batchsize, 1, 1+win_rad*2, 1+win_rad*2)
        assert c1_batch.shape == (batchsize, 1, 1, 1)
        assert torch.all(c1_batch != 0)
        assert torch.all(torch.isfinite(c1_batch))
        if not torch.all(torch.isfinite(rho_batch)):
            print("rho_batch:", rho_batch)
        assert torch.all(torch.isfinite(rho_batch))
        assert torch.all(rho_batch[:, 0, win_rad, win_rad] > 0)

@pytest.mark.order(1)
def test_get_train_nowindow_dls(test_res):
    folder = "tests/test_data"
    batchsize = 32
    split = 0.7
    shuffle = True
    num_workers = 1

    train_dl, _, rho_profiles, c1_profiles = get_dls(folder, batchsize, split, shuffle, num_workers)
    assert len(train_dl) > 0
    assert len(rho_profiles) > 0
    assert len(c1_profiles) > 0
    assert len(rho_profiles) == len(c1_profiles)