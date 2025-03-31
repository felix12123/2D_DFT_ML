import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import numpy as np
import json


from config import TOTAL_MEMORY

from ._simfolder_utils import max_file_num

# estimete how many batches should be created to be less than TOTAL_MEMORY in size
def estimate_num_batches(datafolder, batchsize, traintest_split=0.8, L=10, dx=0.5, win_rad_bins=None):
    """
    Estimate a reasonable number of batches to be created to be less than TOTAL_MEMORY in size.
    """
    if win_rad_bins is None:
        win_rad_bins = int(np.ceil(2 / dx))
    # count number of profiles in datafolder
    num_profiles = max_file_num(datafolder)
    num_train_profiles = int(num_profiles * traintest_split) * 4
    num_windows = num_train_profiles * (L // dx) * (L // dx)
    num_windows = int(num_windows)
    percent_used = min(1, TOTAL_MEMORY / (num_windows * 4 * ((2*win_rad_bins+1)**2 + 1)))
    num_batches = int(num_windows * percent_used / batchsize)
    return num_batches


# define class to contain the model, optimizer, loss function, metric function etc
class MLTraining:
    def __init__(self, folder, model, L, dx, datafolder, criterion=nn.MSELoss(), optimizer_class=torch.optim.Adam, metric=nn.L1Loss(), trained_epochs=0, windowed=True, batchsize=64, shuffle=True, num_workers=4, traintest_split=0.8, save_interval=1, lr_decay=None, batch_lr_decay=1-2e-5, lr_start=3e-4, min_lr=0, device=None, total_batches=None):
        """
        Initialize the MLTraining class. This class contains the model, optimizer, loss function, metric function, and other parameters for training the model.
        It also contains methods for training the model, logging the training progress, and saving the model.
        This is the central class of this project.
        It can be trained with mlt.train(epochs) and the predictions can be visualized with mlt.show_c1_prediction().
        The iterated density profiles can be visualized with mlt.demo_model_iteration().
        The class can be saved to the directory path mlt.workspace with mlt.save() and loaded with load_MLTraining(folder).
        Args:
            folder (str): The folder where the model and training data are stored.
            model (nn.Module): The model to be trained. Choose from the models in src.training.models.
            L (float): The size of the simulation box in the simulations of the datafolder.
            dx (float): The grid spacing of the simulations.
            datafolder (str): The folder where the simulated training data is stored.
            criterion (nn.Module): The loss function to be used.
            optimizer_class (torch.optim.Optimizer): The optimizer class to be used.
            metric (nn.Module): The metric function to be used.
            trained_epochs (int): The number of epochs the model has already been trained for.
            windowed (bool): Whether to use windowed training or not (It is highly recommended to do so).
            batchsize (int): The batch size to be used.
            shuffle (bool): Whether to shuffle the training data or not (True is highly recommended).
            num_workers (int): The number of workers to be used for data loading.
            traintest_split (float): The fraction of the data to be used for training.
            save_interval (int): The interval at which to save the model.
            lr_decay (float): The learning rate decay factor. Current LR is multiplied by this factor after each epoch. (LR = LR * lr_decay ** epoch)
            batch_lr_decay (float): The learning rate decay factor per batch. This can alternatively be supplied instead of lr_decay. lr_decay = batch_lr_decay ** total_batches
            lr_start (float): The initial learning rate.
            min_lr (float): The minimum learning rate.
            device (str): The device to be used for training (e.g. "cuda" or "cpu").
            total_batches (int): The total number of batches to be used for training.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not os.path.exists(datafolder):
            raise ValueError("Data folder does not exist")
            
        ### default values ###
        # check if model is constructor
        if callable(model) and not isinstance(model, nn.Module):
            try:
                model = model(int(np.ceil(1/dx)), 20)
            except TypeError:
                raise TypeError("The provided model is not a constructor")
        if not isinstance(model, nn.Module):
            raise TypeError("The provided model is not an instance of nn.Module")
        if total_batches is None:
            total_batches = estimate_num_batches(datafolder, batchsize, traintest_split, L, dx)
        if lr_decay is None:
            if batch_lr_decay is None:
                TypeError("Either lr_decay or batch_lr_decay must be provided")
            else:
                lr_decay = batch_lr_decay ** total_batches
        
        # initialize workspace structure
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "records"), exist_ok=True)
        os.makedirs(os.path.join(folder, "plots"),   exist_ok=True)
        
        

        ### define class variables ###
        self.workspace = folder
        self.datafolder = datafolder
        self.model_savepath = self.workspace + "/model.pt"
        self.optimizer_savepath = self.workspace + "/optimizer.pt"
        self.scheduler_savepath = self.workspace + "/scheduler.pt"
        self.train_logs_savepath = self.workspace + "/records/train_logs.csv"
        self.test_logs_savepath = self.workspace + "/records/test_logs.csv"
        
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(model.parameters(), lr=lr_start)
        self.scheduler_class = lrs.ExponentialLR
        self.scheduler = self.scheduler_class(self.optimizer, gamma=lr_decay)
        self.metric = metric
        
        self.L = L
        self.dx = dx
        self.windowed = windowed
        self.batchsize = batchsize
        self.total_batches = total_batches
        self.shuffle = shuffle
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.traintest_split = traintest_split
        self.num_workers = num_workers
        self.save_interval = save_interval
        self.trained_epochs = trained_epochs
        

    ### import methods ###
    from ._training import train, log_kernels_tb, log_train_progress, log_test_progress, add_hist_tb, calc_criterion, train_smooth
    from ._dl import create_dataloader # create a dataloader for the training data
    from ._plotting import show_kernels # show the kernels of the first convolutional layer
    from ._minimize import minimize # minimize for an external potential using the model
    from ._demo import demo_model_iteration, show_c1_prediction, create_model_iteration_profiles, demo_model_iteration2
    from ._bench import calc_loss, bench_c1, bench_iter_time, bench_iter
    from ._gr import g_r
    from ._storing import save
    from ._model_utils import smooth_kernel
    from ._iter_folders import ensure_model_iter_existance
    
    def get_tb_folder(self):
        # get the folder for the tensorboard logs
        return os.path.join(self.workspace, "tb_logs")
    def set_workspace(self, folder):
        # set the workspace to a new folder
        self.workspace = folder
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(os.path.join(self.workspace, "records"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace, "plots"),   exist_ok=True)
        self.model_savepath = self.workspace + "/model.pt"
        self.optimizer_savepath = self.workspace + "/optimizer.pt"
        self.scheduler_savepath = self.workspace + "/scheduler.pt"
        self.train_logs_savepath = self.workspace + "/records/train_logs.csv"
        self.test_logs_savepath = self.workspace + "/records/test_logs.csv"

        self.save()
    


def load_MLTraining(workspace: str, device=None) -> 'MLTraining':
    """
    Load an MLTraining object from a folder containing the saved parameters and model states.

    Args:
        workspace (str): The folder where the parameters and model states are saved.
        device (str, optional): The device to load the model onto. Defaults to None.

    Returns:
        MLTraining: The reconstructed MLTraining object.
    """
    # Paths to the saved files
    params_path = os.path.join(workspace, "params.json")
    model_path = os.path.join(workspace, "model.pt")
    optimizer_path = os.path.join(workspace, "optimizer.pt")
    scheduler_path = os.path.join(workspace, "scheduler.pt")

    # Check if the required files exist
    if not os.path.exists(workspace):
        raise FileNotFoundError(f"Workspace folder not found: {workspace}")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model state file not found: {model_path}")
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError(f"Optimizer state file not found: {optimizer_path}")
    if not os.path.exists(scheduler_path):
        raise FileNotFoundError(f"Scheduler state file not found: {scheduler_path}")

    # Load parameters from the JSON file
    with open(params_path, "r") as json_file:
        params = json.load(json_file)

    # Set the device if not provided
    if device is None:
        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically create the model using the model type from the JSON file
    model_class = getattr(__import__('src.training.models', fromlist=[params["model"]]), params["model"])
    # check if convtype string is a nn module and load it if it is
    if params["convtype"] in dir(nn):
        convtype = getattr(nn, params["convtype"])
    else:
        convtype = getattr(__import__('src.training.models', fromlist=[params["convtype"]]), params["convtype"])
    # check if activation string is a nn module and load it if it is
    if params["activation"] in dir(nn):
        activation = getattr(nn, params["activation"])
    else:
        activation = getattr(__import__('src.training.models', fromlist=[params["activation"]]), params["activation"])

    model = model_class(params["Nd"], params["Nn"], hidden_channels=params["phi_width"], hidden_layers=params["phi_layers"], convtype=convtype, activation=activation)

    # Reconstruct the MLTraining instance
    instance = MLTraining(
        folder=workspace,
        model=model,
        L=params["L"],
        dx=params["dx"],
        datafolder=params["datafolder"],
        criterion=getattr(nn, params["criterion"])(),
        optimizer_class=getattr(torch.optim, params["optimizer_class"]),
        metric=getattr(nn, params["metric"])(),
        trained_epochs=params["trained_epochs"],
        windowed=params["windowed"],
        batchsize=params["batchsize"],
        shuffle=params["shuffle"],
        num_workers=params["num_workers"],
        traintest_split=params["traintest_split"],
        save_interval=params["save_interval"],
        lr_decay=params["lr_decay"],
        lr_start=params["lr_start"],
        min_lr=params["min_lr"],
        device=device,
        total_batches=params["total_batches"]
    )

    # Load the model state
    instance.model.load_state_dict(torch.load(model_path, map_location=device))
    instance.model = instance.model.to(device)

    # Load the optimizer state
    instance.optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    # Load the scheduler state
    instance.scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))

    return instance

