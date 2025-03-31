import json
import torch
import os
import torch.nn as nn
import numpy as np


from .models import*
from typing import TYPE_CHECKING

# to avoid circular imports but still allow type checking
# this is not necessary for the code to run, but it helps with autocompletion
if TYPE_CHECKING: # pragma: no cover
    from . import MLTraining

def save_other_params_to_json(self: 'MLTraining', filepath: str):
    # Collect all relevant parameters into a dictionary
    if hasattr(self.model, "omegas"):
        convtype = self.model.omegas.__class__.__name__
    else:
        convtype = self.model.omegas1.__class__.__name__
    params = {
        "model": self.model.__class__.__name__,  # Save model class name
        "L": self.L,
        "dx": self.dx,
        "datafolder": self.datafolder,
        "criterion": self.criterion.__class__.__name__,
        "optimizer_class": self.optimizer_class.__name__,
        "scheduler_class": self.scheduler_class.__name__,
        "metric": self.metric.__class__.__name__,
        "trained_epochs": self.trained_epochs,
        "windowed": self.windowed,
        "batchsize": self.batchsize,
        "shuffle": self.shuffle,
        "num_workers": self.num_workers,
        "traintest_split": self.traintest_split,
        "savepath": self.model_savepath,
        "save_interval": self.save_interval,
        "lr_decay": self.lr_decay,
        "lr_start": self.optimizer.param_groups[0]["lr"],
        "Nn": self.model.Nn,
        "Nd": self.model.Nd,
        "phi_layers": len(self.model.phi)//2,
        "phi_width": self.model.phi[0].weight.shape[0],
        "convtype": convtype,
        "activation": self.model.phi[1].__class__.__name__,
        "min_lr": self.min_lr,
        "total_batches": self.total_batches
    }

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    # Write the parameters to a JSON file
    with open(filepath, "w") as json_file:
        json.dump(params, json_file, indent=4)

def save(self:'MLTraining'):
    padding_was_zero = False
    if self.model.is_padding_zero():
        self.model.add_padding()
        padding_was_zero = True
    torch.save(self.model.state_dict(), self.model_savepath)
    torch.save(self.optimizer.state_dict(), self.optimizer_savepath)
    torch.save(self.scheduler.state_dict(), self.scheduler_savepath)

    # save the other parameters as a json file
    save_other_params_to_json(self, os.path.join(self.workspace, "params.json"))
    
    if padding_was_zero:
        self.model.remove_padding()



