
TOTAL_MEMORY = 4 * 2**30  # x * 1 GB

# Good Parameters for training with sin potential simulation data
SIN_TRAIN_PARAMS = {
    "Nn": 8,
    "lrs": 3e-5,
    "lrd": 1 - 5e-3,
    "bs": 64,
    "epochs": 450
}

# Good Parameters for training with plateau + box + sin potential simulation data
PLAT_TRAIN_PARAMS = {
    "Nn": 8,
    "lrs": 7e-5,
    "lrd": 1 - 5e-3,
    "bs": 96,
    "epochs": 500
}
