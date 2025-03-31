from src.training import load_MLTraining

import os
import pytest

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.order(3)
def test_g_r(test_res):
    mlt_instance = load_MLTraining("tests/fmt_train")
    mlt_instance.g_r(mus=[0, 1, 2, 3, 4], plot_save_path=f"{mlt_instance.workspace}/plots/g_r_test.pdf", eps=1e-5)
    assert os.path.exists(f"{mlt_instance.workspace}/plots/g_r_test.pdf")
    