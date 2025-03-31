from src.training import *
from src.training._dl import *
from src.training._demo import *
from src.training.models import Model_FMT, Model_Bay
from shutil import copyfile
import os
def get_gpu_usage():
    return float(os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv").read().split("\n")[1].split(" ")[0])


print("\n\n")




