from src.training import load_MLTraining

import pytest


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(4)
def test_bench_iter(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    losses, metrics, meanrhos = mlt_instance.bench_iter(max_iter=200)
    assert all(losses > 0)
    assert all(losses < 1e-2)
    assert all(metrics > 0)
    assert all(metrics < 1e-1)
    assert all(meanrhos > 0)
    

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(4)
def test_bench_iter_time(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    time_mean, time_std = mlt_instance.bench_iter_time(iters=300, repetitions=3, return_std=True)
    assert time_mean > 0
    assert time_std > 0
    assert time_mean < 1e2
    assert time_std < 1e2
    

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(4)
def test_bench_c1(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    losses, metrics = mlt_instance.bench_c1()
    assert all(losses > 0)
    assert all(losses < 1e-2)
    assert all(metrics > 0)
    assert all(metrics < 1e-1)