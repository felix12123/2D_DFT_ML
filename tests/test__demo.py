import pytest

from src.training import *
from src.training._demo import *


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(3)
def test_demo_model_iteration1(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    mlt_instance.demo_model_iteration(max_iter=100, eps=1e-4)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(3)
def test_demo_model_iteration2(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    mlt_instance.demo_model_iteration2(
        max_iter=100, eps=1e-4, fmt_folder="tests/fmt_res")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(3)
def test_show_c1_prediction(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    mlt_instance.show_c1_prediction()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(3)
def test_check_simulation_accuracy(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    check_simulation_accuracy(mlt_instance.datafolder, mlt_instance.L)
