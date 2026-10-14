import os, sys

# ---- First pass: install correct versions, then force a kernel restart ----
if "facenet_pytorch" not in sys.modules:
    try:
        import facenet_pytorch
        need_install = False
    except ImportError:
        need_install = True

    import subprocess
    result = subprocess.run(["pip", "show", "pillow"], capture_output=True, text=True)
    pillow_ok = "Version: 10.2.0" in result.stdout

    if not pillow_ok or need_install:
        print("Installing correct package versions, then restarting the kernel automatically...")
        os.system("pip uninstall -y pillow -q")
        os.system('pip install "pillow==10.2.0" --no-deps -q')
        os.system("pip install facenet-pytorch --no-deps -q")
        print("Restarting now — after the kernel restarts, just run this SAME cell again to continue into training.")
        os.kill(os.getpid(), 9)  # forces an automatic kernel restart

# ---- Second pass (after the auto-restart): everything below runs normally ----
print("Environment already correct — proceeding to training...")

import os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from facenet_pytorch import InceptionResnetV1
from datasets import load_dataset
from google.colab import drive

# ... [rest of the v6 training code exactly as before] ...
