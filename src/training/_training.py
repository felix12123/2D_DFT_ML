import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from time import time
from torch.utils.tensorboard import SummaryWriter
from .models import*
import numpy as np
from typing import TYPE_CHECKING

# to avoid circular imports but still allow type checking
# this is not necessary for the code to run, but it helps with autocompletion
if TYPE_CHECKING: # pragma: no cover
    from . import MLTraining

def np_to_tensor(data):
    return torch.from_numpy(data).reshape(1, 1, data.shape[0], data.shape[1]).float()

def tensor_to_np(data):
    return data.detach().squeeze().cpu().numpy()

def calc_criterion(self:'MLTraining', rho_profiles, c1_profiles, criterion, device):
    losses = []
    self.model.to(device)
    self.model.eval()
    if self.windowed:
        self.model.add_padding()
    with torch.no_grad():
        for rho, c1 in zip(rho_profiles, c1_profiles):
            rho, c1 = np_to_tensor(rho).to(device), np_to_tensor(c1).to(device)
            outputs = self.model(rho)
            
            dens_mask = torch.flatten(rho) > 0
            loss = criterion(torch.flatten(outputs)[dens_mask], torch.flatten(c1)[dens_mask])
            losses.append(loss.item())
    if self.windowed:
        self.model.remove_padding()
    self.model.train()
    return np.array(losses)



def iteration_test(self, writer, epoch, rho_profiles, c1_profiles, device):
    # to be implemented
    return



def log_train_progress(self, writer, epoch, loss, metric):
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Metric/train', metric, epoch)
    writer.add_scalar('Learning rate', self.optimizer.param_groups[0]["lr"], epoch)
    # add row to train_logs_savepath
    if not os.path.exists(self.train_logs_savepath):
        with open(self.train_logs_savepath, "w") as f:
            f.write("epoch,loss,metric,lr,time\n")
    with open(self.train_logs_savepath, "a") as f:
        # f.write("%d,%.6e,%.6e,%.6e\n" % (epoch, loss, metric, self.optimizer.param_groups[0]["lr"], ))
        f.write("%d,%.6e,%.6e,%.6e,%.10e\n" % (epoch, loss, metric, self.optimizer.param_groups[0]["lr"], time()))

def log_test_progress(self, writer, epoch, rho_profiles, c1_profiles, device):
    N = len(rho_profiles)
    N_train = int(N*self.traintest_split)
    N_train = min(N_train, N-1)
    N_test = N - N_train
    
    test_loss = self.calc_criterion(rho_profiles[-N_test:], c1_profiles[-N_test:], self.criterion, device).mean()
    test_metric = self.calc_criterion(rho_profiles[-N_test:], c1_profiles[-N_test:], self.metric, device).mean()
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Metric/test', test_metric, epoch)
    
    # add row to test_logs_savepath
    if not os.path.exists(self.test_logs_savepath):
        with open(self.test_logs_savepath, "w") as f:
            f.write("epoch,loss,metric,lr,time\n")
    with open(self.test_logs_savepath, "a") as f:
        f.write("%d,%.6e,%.6e,%.6e,%.12e\n" % (epoch, test_loss, test_metric, self.optimizer.param_groups[0]["lr"], time()))
    
    

def log_kernels_tb(self:'MLTraining', writer, epoch):
    if hasattr(self.model, 'collect_all_kernels') and hasattr(self.model, "omegas"):
        writer.add_figure('Kernels/omegas', self.show_kernels(conv_layer=0, output_file=""), epoch)
    if hasattr(self.model, 'collect_all_kernels') and hasattr(self.model, "omegas1") and hasattr(self.model, "omegas2"):
        writer.add_figure('Kernels/conv1', self.show_kernels(conv_layer=0, output_file=""), epoch)
        writer.add_figure('Kernels/conv2', self.show_kernels(conv_layer=1, output_file=""), epoch)
    

def add_tensorboard_info(self, writer, device):
    writer.add_graph(self.model, torch.randn(1, 1, int(np.ceil(self.L/self.dx)), int(np.ceil(self.L/self.dx))).to(device))
    writer.add_text('model', str(self.model), 0)
    writer.add_text('optimizer', str(self.optimizer), 0)
    writer.add_text('scheduler', str(self.scheduler), 0)
    writer.add_text('criterion', str(self.criterion), 0)
    writer.add_text('metric', str(self.metric), 0)
    writer.add_text('training', str(self), 0)    

def add_hist_tb(self, writer, epoch, rho_profiles, c1_profiles, device):
    # create histogram data (loss for each profile)
    hist_data = self.calc_criterion(rho_profiles, c1_profiles, self.criterion, device)
    writer.add_histogram('LogLoss/all', np.log10(np.array(hist_data)), epoch)


def train(self:'MLTraining', epochs: int, loglevel=0):
    if epochs <= 0:
        if loglevel <= 1: print("no epochs to train")
        return self.model, 0

    if loglevel <= 1: print("start training of following model with data from folder %s for %d epochs" %
              (self.datafolder, epochs))
    if os.path.dirname(self.model_savepath) != "":
        os.makedirs(os.path.dirname(self.model_savepath), exist_ok=True)
    
    device = self.device
    
    # Create a tensorboard writer
    writer_path = self.get_tb_folder()
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(writer_path, flush_secs=30)
    if loglevel <= 1: print("writer path: ", writer_path)
    

    ##### MODEL LOADING #####
    # load new model
    if not isinstance(self.model, nn.Module) and callable(self.model):  # if model is constructor
        self.model = self.model(int(np.ceil(1/self.dx)))

    self.model = self.model.to(device)
    if self.trained_epochs == 0: add_tensorboard_info(self, writer, device)
    if loglevel <= 0: print(self.model)
    if loglevel <= 1: print("total parameters: %d" % sum(p.numel()
              for p in self.model.parameters()))

    # change model padding to zero to use windows
    if self.windowed:
        self.model.remove_padding()

    ##### DATA LOADER #####
    if loglevel <= 1: print("loading data...", end="")
    trainloader, rho_profiles, c1_profiles = self.create_dataloader()
    
    if loglevel <= 1: print(" done! %d training samples" %
              (len(trainloader.dataset)))
    if loglevel <= 0: print("size of windows: %dx%d" % (
            trainloader.dataset[0][0].shape[1], trainloader.dataset[0][0].shape[2]))
    if loglevel <= 1: print("number of batches in training set: %d" % len(trainloader))

    ##### TRAINING #####
    if loglevel <= 1: print("start training...")

    training_start_time = time()
    if loglevel <= 2: print("LR will be %.2e at the end of training" % (self.optimizer.param_groups[0]["lr"] * self.lr_decay ** epochs))
    for epoch in range(self.trained_epochs+1, self.trained_epochs+epochs+1):
        running_loss = 0.0
        running_metric = 0.0
        count = 0
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Print device information

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(torch.flatten(
                outputs), torch.flatten(labels))
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.model.apply_weight_constraint()
            running_metric += self.metric(outputs, labels).item()
            count += 1

        if training_start_time is not None:
            if loglevel <= 2: print("Training will approximately take %.2f hours" % ((time()-training_start_time) * epochs / 3600))
            training_start_time = None


        ### TESTING AND LOGGING ###
        if loglevel <= 2: print('[epoch %.3d]   lr: %.2e   loss: %.2e   metric: %.2e'
                  % (epoch,
                     self.optimizer.param_groups[0]["lr"],
                     running_loss / count, running_metric / count),
                  end="   ")
        if self.optimizer.param_groups[0]["lr"] >= self.min_lr: self.scheduler.step()
        
        if epoch % 5 == 0:
            t_start_test = time()
            self.log_test_progress(writer, epoch, rho_profiles, c1_profiles, device)
            if loglevel <= 2: print("   testtime: %.2fs" % (time()-t_start_test), end="")
        if epoch % 20 == 0:
            t_start_test = time()
            self.add_hist_tb(writer, epoch, rho_profiles, c1_profiles, device)
            if loglevel <= 2: print("   histtime: %.2fs" % (time()-t_start_test), end="")
            self.log_kernels_tb(writer, epoch)
        
        # write loss to tensorboard
        self.log_train_progress(writer, epoch, running_loss / count,
                       running_metric / count)

        self.trained_epochs += 1
        if loglevel <= 2: print()
        self.save()
    
    self.save()
    writer.flush()
    writer.close()
    if loglevel <= 2: print("finished training!")

    return self.model, 0



def train_smooth(self, epochs:list, smooths:list, loglevel=0):
    assert len(epochs) == len(smooths) + 1, "Number of epochs must be one more than number of smooths"
    if self.trained_epochs >= np.sum(epochs):
        return
    if loglevel <= 3: print("Total number of epochs: %d" % sum(epochs)); print('Trained so far: %s' % self.trained_epochs)
    for i in range(len(epochs)-1):
        target_epochs = sum(epochs[:i+1])
        if target_epochs <= self.trained_epochs:
            continue
        if loglevel <= 3: print("Training %d epochs with smooth %.2f" % (epochs[i], smooths[i]))
        self.train(epochs[i], loglevel=loglevel)
        self.smooth_kernel(smooths[i])
    self.train(sum(epochs)-self.trained_epochs, loglevel=loglevel)
        
        
    
    
        