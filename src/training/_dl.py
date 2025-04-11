# from . import MLTraining
from ._pot import evaluate_potential, load_potential_params

import os
import json
import numpy as np
import torch
import config
from ._simfolder_utils import *

TOTAL_MEMORY = config.TOTAL_MEMORY
from typing import TYPE_CHECKING
if TYPE_CHECKING: # pragma: no cover
    from . import MLTraining

def create_dataloader(self:'MLTraining'):
    if self.windowed:
        return get_train_window_dls(self.datafolder, int(np.ceil(2/self.dx)), self.total_batches, self.batchsize, self.traintest_split, self.shuffle, num_workers=self.num_workers)
    else:
        dls = get_dls(self.datafolder, self.batchsize, self.traintest_split, self.shuffle, num_workers=self.num_workers)
        return dls[0], dls[2], dls[3]

def get_dls(folder: str, batchsize: int = 8, split: float = 0.9, shuffle: bool = True, num_workers: int = 4):
    if not 0 <= split <= 1:
        raise ValueError('split must be between 0 and 1')
    rho_profiles = []
    c1_profiles = []
    for i in range(1, max_file_num(folder)+1):
        rho_profiles.append(get_rho(folder, i))
        c1_profiles.append(get_c1(folder, i, rho_profiles[-1]))

    if len(rho_profiles) == 0:
        raise ValueError('No data files found in folder')
    elif len(rho_profiles) == 1:
        print("Only one file found, no test set will be created")
        split = 1.0

    N = len(rho_profiles)
    NL = len(rho_profiles[0])
    N_train = int(split * N)

    train_data = torch.from_numpy(np.float32(
        np.stack(rho_profiles[:N_train], axis=0)).reshape(N_train, 1, NL, NL))
    train_label = torch.from_numpy(np.float32(
        np.stack(c1_profiles[:N_train], axis=0)).reshape(N_train, 1, NL, NL))
    train_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        train_data, train_label), batch_size=batchsize, shuffle=shuffle, num_workers=num_workers)
    if split == 1.0:
        return train_dl, None
    
    test_data = torch.from_numpy(np.float32(
        np.stack(rho_profiles[N_train:], axis=0)).reshape(N-N_train, 1, NL, NL))
    test_label = torch.from_numpy(np.float32(
        np.stack(c1_profiles[N_train:], axis=0)).reshape(N-N_train, 1, NL, NL))
    test_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        test_data, test_label), batch_size=batchsize, shuffle=False, num_workers=num_workers)

    return train_dl, test_dl, rho_profiles, c1_profiles

def get_valid_windows(rho, c1):
    return np.where((rho > 0) & np.isfinite(c1))
def get_num_valid_windows(rho:np.ndarray, c1:np.ndarray):
    return np.sum((rho > 0) & np.isfinite(c1))
# for lists of numpy arrays
def get_num_valid_windows_all(rhos:list, c1s:list):
    return sum([get_num_valid_windows(rho, c1) for (rho, c1) in zip(rhos, c1s)])

def add_flipped_versions(rho_profiles, c1_profiles):
    rho_profiles_flipped = np.concatenate([rho_profiles, np.flipud(rho_profiles), np.fliplr(rho_profiles), np.flipud(np.fliplr(rho_profiles))])
    c1_profiles_flipped = np.concatenate([c1_profiles, np.flipud(c1_profiles), np.fliplr(c1_profiles), np.flipud(np.fliplr(c1_profiles))])
    return rho_profiles_flipped, c1_profiles_flipped
    


def get_train_window_dls(folder: str, win_rad: int, total_batches: int, batchsize: int = 256, split: float = 0.9, shuffle: bool = True, percent_used: float=None, num_workers: int = 4):
    N = max_file_num(folder)
    N_train = int(np.round(N*split))
    if N_train >= N:
        N_train = N - 1
    if N_train <= 0:
        raise ValueError("Not enough files in folder to create training set")
    
    rho_profiles_original = []
    c1_profiles_original = []
    for i in range(1, N+1):
        rho = get_rho(folder, i)
        c1 = get_c1(folder, i, rho)
        rho_profiles_original.append(rho)
        c1_profiles_original.append(c1)
    # we want mirrored versions of each profile in the profile lists
    rho_profiles, c1_profiles = add_flipped_versions(rho_profiles_original[:N_train], c1_profiles_original[:N_train])
    

    L = len(rho_profiles[0])
    
    num_valid_windows = np.array([get_num_valid_windows_all(rho, c1) for (rho, c1) in zip(rho_profiles, c1_profiles)])
    

    if percent_used == None:
        windows_needed = total_batches * batchsize
        percent_used = windows_needed / np.sum(num_valid_windows)
        percent_used = min(1, percent_used)

    windows_per_file = np.ceil(num_valid_windows * percent_used).astype(int)

    window_container_train = np.zeros((sum(windows_per_file), 1, 1+2*win_rad,1+2*win_rad), np.float32)
    c1_container_train = np.zeros((sum(windows_per_file),1,1,1), np.float32)
    
    current_index = 0
    padded_rho = np.zeros((L+win_rad*2, L+win_rad*2), np.float32)
    for n in range(len(rho_profiles)):
        padded_rho = np.pad(rho_profiles[n], win_rad, 'wrap') # pad rho with periodic boundary conditions
        
        # determine which windows to use via shuffeling
        valid_indexes = get_valid_windows(rho_profiles[n], c1_profiles[n]) # get all valid coordinates
        selected_indexes = list(range(num_valid_windows[n])) # create list of indexes
        np.random.shuffle(selected_indexes) # shuffle the indexes
        selected_indexes = selected_indexes[:windows_per_file[n]] # select the first windows_per_file indexes
        for i in selected_indexes:
            x = valid_indexes[0][i]
            y = valid_indexes[1][i]
            window_container_train[current_index, 0, :,:] = padded_rho[x:x+2*win_rad+1, y:y+2*win_rad+1]
            c1_container_train[current_index, 0, 0, 0] = c1_profiles[n][x,y]
            current_index += 1
    
    window_container_train = window_container_train[:current_index]
    c1_container_train = c1_container_train[:current_index]
    
    assert all(np.isfinite(c1_container_train))
    assert all(c1_container_train != 0)
    assert all(window_container_train[:, :, win_rad, win_rad] > 0) # check that the center of the window is not zero
    
    # create DataLoader objects
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(window_container_train), torch.from_numpy(c1_container_train))
    
    # cut data to fit the number of batches total_batches
    train_data.tensors = (train_data.tensors[0][:total_batches*batchsize], train_data.tensors[1][:total_batches*batchsize])    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, prefetch_factor=4)
    
    return train_loader, rho_profiles_original, c1_profiles_original