from src.training._dl import *
import pytest

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
    assert len(train_dl) > 0
    assert len(rho_profiles) > 0
    assert len(c1_profiles) > 0
    assert len(rho_profiles) == len(c1_profiles)
    assert len(train_dl) == total_batches

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