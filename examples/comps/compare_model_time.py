import torch
import numpy as np
import matplotlib.pyplot as plt
from src.training._demo import get_rho, max_file_num
from src.training._training import np_to_tensor
import src.training.models as _models
import os
from time import time


Nn = 8
Nd = 20
ms = [_models.Model_FMT2(Nd, Nn), _models.Model_FMT_one_conv(Nd, Nn), _models.Model_FMT2_small(Nd, Nn)]
m_names = ["Model_FMT2", "Model_FMT_one_conv", "Model_FMT2_small"]



datafolder = "/share/train_data/dx005-1e11s-sin"


rhos = [np_to_tensor(get_rho(datafolder, i)).to('cuda') for i in range(1, max_file_num(datafolder)+1)]

def get_model_speed_train(m, rhos):
    times = []
    m.to('cuda')
    m.train()
    m(rhos[0])
    for _ in range(10):
        for rho in rhos:
            t0 = time()
            m(rho)
            times.append(time() - t0)
    return np.median(times), np.std(times)
def get_model_speed_work(m, rhos):
    times = []
    m.to('cuda')
    m.eval()
    m(rhos[0])
    with torch.no_grad():
        for _ in range(10):
            for rho in rhos:
                t0 = time()
                m(rho)
                times.append(time() - t0)
    plt.scatter(np.arange(len(times)), times)
    plt.savefig("del")
    return np.median(times), np.std(times)


time_work = []
time_work_std = []
time_train = []
time_train_std = []
for i in range(len(ms)):
    avgt, stdt = get_model_speed_work(ms[i], rhos)
    time_work.append(avgt / rhos[0].shape[3]**2)
    time_work_std.append(stdt / rhos[0].shape[3]**2)
    
    avgt, stdt = get_model_speed_train(ms[i], rhos)
    time_train.append(avgt / rhos[0].shape[3]**2)
    time_train_std.append(stdt / rhos[0].shape[3]**2)
    

print("Model\t\tWork\tTrain")
for i in range(len(ms)):
    print(f"{m_names[i]}\t{time_work[i]}\t{time_train[i]}")

# plot results
plotfolder = "media/Model_speed"
os.makedirs(plotfolder, exist_ok=True)


plt.figure(figsize=(5,5), dpi=300, facecolor=(1,1,1,0))
plt.bar(np.arange(len(m_names)) - 0.2, time_work, label="Work", width=0.4)
plt.errorbar(np.arange(len(m_names)) - 0.2, time_work, yerr=time_work_std, fmt='none', color='black', capsize=2, capthick=1, elinewidth=1)
plt.bar(np.arange(len(m_names)) + 0.2, time_train, label="Train", width=0.4)
plt.errorbar(np.arange(len(m_names)) + 0.2, time_train, yerr=time_train_std, fmt='none', color='black', capsize=2, capthick=1, elinewidth=1)
plt.legend()

plt.xticks(np.arange(len(m_names)), m_names)
plt.xlabel("Model")
plt.ylabel("Time per application per number in profile [s]")
plt.title(f"Model speed for {rhos[0].shape[3]} x {rhos[0].shape[3]} input profiles")

plt.savefig(os.path.join(plotfolder, "Model_speed.png"))
plt.close()






