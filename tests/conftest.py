# filepath: /home/felix/VSC_Repos/2D_ML_DFT/tests/conftest.py
import os
import pytest
import numpy as np
import shutil
from src.create_df import create_td

@pytest.fixture(scope="session")
def test_res():
    """Creates a test reservoir before the test and deletes it afterward."""
    
    res_path = "tests/test_data"

    # Setup: Create a file
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(res_path + "/unc", exist_ok=True)
    os.makedirs(res_path + "/rho", exist_ok=True)
    os.makedirs(res_path + "/pot", exist_ok=True)
    for i in range(1, 11):
        np.savetxt(res_path + f"/unc/rhostd_{i}.csv", np.random.rand(200, 200)/100, delimiter=",")
        np.savetxt(res_path + f"/rho/rho_{i}.csv", np.ones((200, 200)), delimiter=",")
        shutil.copy("tests/data/potential_1.json", res_path + f"/pot/potential_{i}.json")
        
    
    fmt_res_path = "tests/fmt_res"
    create_td(fmt_res_path, 8, 1e-6)
    
    
    # save msg to file
    # with open("tests/ls_output.txt", "w") as f:
    #     f.write(msg)
    
    
    yield res_path  # Provide file path to the test

    # Teardown: Delete the directory
    if os.path.exists(res_path):
        print(f"Cleaning up: {res_path}")
        shutil.rmtree(res_path)
    if os.path.exists(fmt_res_path):
        print(f"Cleaning up: {fmt_res_path}")
        # print("Cleaning up: ", fmt_res_path)



def delete_tempory_files():
    # Define the paths to clean up
    paths_to_cleanup = [
        "tests/mock_workspace_Bay_nowindows",
        "tests/mock_workspace_Bay",
        "tests/mock_workspace_FMT",
        "tests/mock_workspace_FMT2",
        "tests/test_data",
        "tests/plots",
        "tests/fmt_train",
        "tests/fmt_res",
    ]
    
    for path in paths_to_cleanup:
        if os.path.exists(path):
            print(f"Cleaning up: {path}")
            shutil.rmtree(path)

def pytest_sessionfinish(session, exitstatus):
    """
    Hook to clean up files after the test session has finished.
    """
    delete_tempory_files()
    
    

# run before tests
def pytest_configure(config):
    delete_tempory_files()