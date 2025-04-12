import pytest
import numpy as np

from src.training import *
from src.training._training import *
from src.training.models import Model_FMT, Model_Bay
from config import SIN_TRAIN_PARAMS


def test_np_to_tensor():
    a = np.ones((20, 20))
    b = np_to_tensor(a)
    assert torch.all(torch.eq(b, torch.ones((1, 1, 20, 20)).float()))
    assert b.shape == (1, 1, 20, 20)


def test_tensor_to_np():
    a = torch.ones((1, 1, 20, 20))
    b = tensor_to_np(a)
    assert np.all(b == np.ones((20, 20)))


# train on a small dataset that only contains files with rho=1 and V=-0.5.
# Check if the model converges to the correct value.
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(2)
def test_train_windows_Bay(test_res):
    training_instance = load_MLTraining("tests/mock_workspace_Bay")
    training_instance.train(15, loglevel=100)

    conv_val = training_instance.model(torch.ones(
        (1, 1, 200, 200)).float().to("cuda")).cpu().detach().numpy().mean()
    print("deviation in test_train_windows_Bay", conv_val+0.5)
    assert abs(conv_val + 0.5) < 0.002


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(2)
def test_train_windows_FMT(test_res):
    training_instance = load_MLTraining("tests/mock_workspace_FMT")
    training_instance.train(15, loglevel=100)

    conv_val = training_instance.model(torch.ones(
        (1, 1, 200, 200)).float().to("cuda")).cpu().detach().numpy().mean()
    print("deviation in test_train_windows_FMT", conv_val+0.5)
    assert abs(conv_val + 0.5) < 0.002


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(2)
def test_train_windows_FMT2(test_res):
    training_instance = load_MLTraining("tests/mock_workspace_FMT2")
    training_instance.train(15, loglevel=100)

    conv_val = training_instance.model(torch.ones(
        (1, 1, 200, 200)).float().to("cuda")).cpu().detach().numpy().mean()
    print("deviation in test_train_windows_FMT2", conv_val+0.5)
    assert abs(conv_val + 0.5) < 0.002


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(2)
def test_train_nowindows(test_res):
    training_instance = load_MLTraining("tests/mock_workspace_Bay_nowindows")
    training_instance.train(20, loglevel=100)

    conv_val = training_instance.model(torch.ones(
        (1, 1, 200, 200)).float().to("cuda")).cpu().detach().numpy().mean()
    print("deviation in test_train_nowindows", conv_val+0.5)
    # often less precise, so 0.05 is acceptable
    assert abs(conv_val + 0.5) < 0.05

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(2)
def test_train_smooth(test_res):
    training_instance = load_MLTraining("tests/mock_workspace_Bay")
    training_instance.train_smooth([10, 10, 10, 10], [0.5, 0.5, 0.5], loglevel=100)

    conv_val = training_instance.model(torch.ones(
        (1, 1, 200, 200)).float().to("cuda")).cpu().detach().numpy().mean()
    print("deviation in test_train_smooth", conv_val+0.5)
    # often less precise, so 0.05 is acceptable
    assert abs(conv_val + 0.5) < 0.05
    assert training_instance.trained_epochs == 40

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(2)
def test_real_train(test_res):
    mlt = MLTraining(
        folder="tests/fmt_train",
        model=Model_Bay(int(1/0.05), 8),
        L=10,
        dx=0.05,
        datafolder="tests/fmt_res",
        total_batches=128,
        batchsize=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr_start=SIN_TRAIN_PARAMS["lrs"],
        lr_decay=SIN_TRAIN_PARAMS["lrd"],
        windowed=True,
        traintest_split=1
    )
    mlt.train(100, loglevel=100)
    # read csv mlt.train_logs_savepath with np
    train_logs = np.loadtxt(mlt.train_logs_savepath, delimiter=",", skiprows=1)
    final_loss = train_logs[-1, 1]
    final_metric = train_logs[-1, 2]
    print("final loss", final_loss)
    print("final metric", final_metric)
    assert final_loss < 1e-3
    assert final_metric < 5e-2