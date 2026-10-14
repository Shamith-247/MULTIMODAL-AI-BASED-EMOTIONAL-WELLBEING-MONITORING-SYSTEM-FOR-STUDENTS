# ---- Cell 1 (FIXED) — Install dependencies ----
!pip install -q --upgrade pip setuptools wheel
!pip install -q git+https://github.com/timesler/facenet-pytorch.git --no-deps
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install -q datasets pdf2image
!apt-get -qq install -y poppler-utils

import PIL
print("Pillow version:", PIL.__version__)

# ---- Cell 2 — Imports and configuration ----
import os, json, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from facenet_pytorch import MTCNN, InceptionResnetV1
from datasets import load_dataset

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "| device:", device)
print("facenet_pytorch imported successfully")
