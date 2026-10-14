# ============================================================
# ResEmoteNet-style — CNN + Squeeze-Excitation + Residual blocks on RAF-DB
# Reconstructed from the published architecture description (IEEE SPL '24)
# ============================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from google.colab import drive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
SAVE_PATH = "/content/drive/MyDrive/signalwell-models/resemotenet_style_best.pt"

# ---- Dataset: same RAF-DB used for v5/v6, for a fair comparison ----
if "face_ds_full" not in dir():
    face_ds_full = load_dataset("deanngkl/raf-db-7emotions")
label_names_face = face_ds_full["train"].features["label"].names
print("Face labels:", label_names_face)

all_labels = face_ds_full["train"]["label"]
idx_train, idx_test = train_test_split(
    np.arange(len(all_labels)), test_size=0.10, random_state=42, stratify=all_labels,
)
face_ds = {"train": face_ds_full["train"].select(idx_train), "test": face_ds_full["train"].select(idx_test)}

train_labels_list = face_ds["train"]["label"]
class_counts = np.bincount(train_labels_list, minlength=len(label_names_face))
class_weights = (1.0 / np.maximum(class_counts, 1))
class_weights = class_weights / class_weights.sum() * len(label_names_face)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ---- Same preprocessing style as the official eval scripts: 64x64, grayscale->3ch, ImageNet norm ----
IMG_SIZE = 64
train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FaceDataset(Dataset):
    def __init__(self, hf_split, train=False):
        self.data = hf_split; self.train = train
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"].convert("RGB")
        return (train_tf if self.train else eval_tf)(img), item["label"]

face_train_ds = FaceDataset(face_ds["train"], train=True)
face_test_ds  = FaceDataset(face_ds["test"], train=False)
BATCH = 64
face_train_loader = DataLoader(face_train_ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
face_test_loader  = DataLoader(face_test_ds, batch_size=128, shuffle=False, num_workers=2)

# ---- Architecture: CNN backbone -> SE block -> 3 residual blocks -> classifier ----
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class ResEmoteNetStyle(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # CNN backbone: 3 conv layers, downsampling
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))   # 64->32
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)) # 32->16
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))# 16->8
        self.se = SEBlock(256)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.se(x)
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        x = self.gap(x)
        return self.classifier(x)

model = ResEmoteNetStyle(len(label_names_face)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS = 25
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=len(face_train_loader), epochs=EPOCHS)
crit = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in face_train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        sched.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)

    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y in face_test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            t_correct += (out.argmax(1) == y).sum().item(); t_total += x.size(0)
    test_acc = t_correct / t_total
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({"state_dict": model.state_dict(), "labels": label_names_face, "test_acc": best_acc}, SAVE_PATH)

    print(f"epoch {epoch+1}/{EPOCHS} loss={running_loss/total:.4f} "
          f"train_acc={correct/total:.4f} test_acc={test_acc:.4f} best={best_acc:.4f}")

print(f"\nResEmoteNet-style best test accuracy: {best_acc:.4f} (saved to {SAVE_PATH})")
