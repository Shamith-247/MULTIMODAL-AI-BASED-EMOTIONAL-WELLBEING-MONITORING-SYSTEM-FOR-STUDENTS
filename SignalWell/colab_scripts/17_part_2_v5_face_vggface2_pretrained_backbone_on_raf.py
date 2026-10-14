# ============================================================
# PART 2 (v5) — FACE: VGGFace2-pretrained backbone on RAF-DB
# Single cell: install -> smoke test -> train -> save direct to Drive
# ============================================================
!pip uninstall -y pillow -q
!pip install "pillow==10.2.0" --no-deps -q
!pip install facenet-pytorch --no-deps -q

import os
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

print("torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
print("GPU available:", torch.cuda.is_available())

# Smoke test — fails fast if there's a cuDNN mismatch, instead of an hour into training
x = torch.randn(8, 3, 160, 160).cuda()
conv = torch.nn.Conv2d(3, 16, 3).cuda()
bn = torch.nn.BatchNorm2d(16).cuda()
print("Smoke test shape:", bn(conv(x)).shape)

print("\n=== [2/3] FACE v5 — VGGFace2-pretrained backbone on RAF-DB ===")

drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
SAVE_PATH = "/content/drive/MyDrive/signalwell-models/face_model_vggface2_rafdb_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "face_ds_full" not in dir():
    face_ds_full = load_dataset("deanngkl/raf-db-7emotions")

label_names_face = face_ds_full["train"].features["label"].names
print("Face labels:", label_names_face)

all_labels = face_ds_full["train"]["label"]
idx_train, idx_test = train_test_split(
    np.arange(len(all_labels)), test_size=0.10, random_state=42, stratify=all_labels,
)
face_ds = {
    "train": face_ds_full["train"].select(idx_train),
    "test":  face_ds_full["train"].select(idx_test),
}

train_labels_list = face_ds["train"]["label"]
class_counts = np.bincount(train_labels_list, minlength=len(label_names_face))
class_weights = (1.0 / np.maximum(class_counts, 1))
class_weights = class_weights / class_weights.sum() * len(label_names_face)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

IMG_SIZE = 160

train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

class FaceDatasetV5(Dataset):
    def __init__(self, hf_split, train=False):
        self.data = hf_split
        self.train = train
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"].convert("RGB")
        tensor = (train_tf if self.train else eval_tf)(img)
        return tensor, item["label"]

face_train_ds = FaceDatasetV5(face_ds["train"], train=True)
face_test_ds  = FaceDatasetV5(face_ds["test"], train=False)

BATCH = 64
face_train_loader = DataLoader(face_train_ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
face_test_loader  = DataLoader(face_test_ds, batch_size=128, shuffle=False, num_workers=2)

backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
face_model = nn.Sequential(
    backbone,
    nn.Dropout(0.4),
    nn.Linear(512, len(label_names_face)),
).to(device)

face_crit = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
best_acc = 0.0

def mixup_batch(x, y, num_classes, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    y_onehot = F.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return x_mixed, y_mixed

def soft_ce(logits, soft_targets, weight=None):
    logp = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * logp).sum(dim=1)
    if weight is not None:
        w = (soft_targets * weight.unsqueeze(0)).sum(dim=1)
        loss = loss * w
    return loss.mean()

@torch.no_grad()
def run_eval(use_tta=True):
    face_model.eval()
    t_correct, t_total = 0, 0
    for x, y in face_test_loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = face_model(x)
            if use_tta:
                x_flip = torch.flip(x, dims=[3])
                out = out.softmax(1) + face_model(x_flip).softmax(1)
        t_correct += (out.argmax(1) == y).sum().item(); t_total += x.size(0)
    return t_correct / t_total

def train_epochs(n_epochs, optimizer, scheduler, phase_name, use_mixup=True):
    global best_acc
    for epoch in range(n_epochs):
        face_model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in face_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if use_mixup and np.random.rand() < 0.5:
                    x_mixed, y_mixed = mixup_batch(x, y, len(label_names_face), alpha=0.2)
                    out = face_model(x_mixed)
                    loss = soft_ce(out, y_mixed, weight=class_weights_t)
                else:
                    out = face_model(x)
                    loss = face_crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None: scheduler.step()
            running_loss += loss.item() * x.size(0)
            with torch.no_grad():
                preds = face_model(x).argmax(1)
            correct += (preds == y).sum().item(); total += x.size(0)
        test_acc = run_eval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"state_dict": face_model.state_dict(), "labels": label_names_face, "test_acc": best_acc},
                       SAVE_PATH)
        print(f"[{phase_name}] epoch {epoch+1}/{n_epochs} loss={running_loss/total:.4f} "
              f"train_acc={correct/total:.4f} test_acc(TTA)={test_acc:.4f} best={best_acc:.4f}")

print("\n-- Phase 1: frozen backbone, head only --")
for p in backbone.parameters(): p.requires_grad = False
phase1_params = [p for p in face_model.parameters() if p.requires_grad]
phase1_opt = torch.optim.AdamW(phase1_params, lr=1e-3)
phase1_sched = torch.optim.lr_scheduler.OneCycleLR(
    phase1_opt, max_lr=1e-3, steps_per_epoch=len(face_train_loader), epochs=4
)
train_epochs(4, phase1_opt, phase1_sched, "phase1-head", use_mixup=False)

print("\n-- Phase 2: unfreeze last blocks --")
unfroze_any = False
for name, p in backbone.named_parameters():
    if any(k in name for k in ["block8", "repeat_3", "last_linear", "last_bn"]):
        p.requires_grad = True
        unfroze_any = True
if not unfroze_any:
    print("WARNING: no matching layer names found")
phase2_params = [p for p in face_model.parameters() if p.requires_grad]
phase2_opt = torch.optim.AdamW(phase2_params, lr=3e-4, weight_decay=1e-4)
phase2_sched = torch.optim.lr_scheduler.OneCycleLR(
    phase2_opt, max_lr=3e-4, steps_per_epoch=len(face_train_loader), epochs=8
)
train_epochs(8, phase2_opt, phase2_sched, "phase2-partial-finetune")

print("\n-- Phase 3: full fine-tune --")
for p in face_model.parameters(): p.requires_grad = True
phase3_opt = torch.optim.AdamW(face_model.parameters(), lr=5e-5, weight_decay=1e-4)
phase3_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    phase3_opt, T_0=len(face_train_loader) * 3, T_mult=1
)
train_epochs(16, phase3_opt, phase3_sched, "phase3-full-finetune")

print(f"\nFACE v5 best test accuracy: {best_acc:.4f} (saved directly to Drive at every new best)")
