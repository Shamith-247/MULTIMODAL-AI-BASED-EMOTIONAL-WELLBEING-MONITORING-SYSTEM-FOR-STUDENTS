# ============================================================
# PART 2 (v7) — FACE: VGGFace2 backbone + CutMix + FIXED EMA + 10-crop TTA
# ============================================================
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

print("torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
x = torch.randn(8, 3, 160, 160).cuda()
print("Smoke test:", nn.BatchNorm2d(16).cuda()(nn.Conv2d(3,16,3).cuda()(x)).shape)

drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
SAVE_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v7_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

IMG_SIZE = 160
train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(0.5), T.RandomRotation(10),
    T.RandomAffine(degrees=0, translate=(0.06,0.06), scale=(0.95,1.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2), T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])
eval_tf = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

class FaceDatasetV7(Dataset):
    def __init__(self, hf_split, train=False):
        self.data = hf_split; self.train = train
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"].convert("RGB")
        return (train_tf if self.train else eval_tf)(img), item["label"]

face_train_ds = FaceDatasetV7(face_ds["train"], train=True)
face_test_ds  = FaceDatasetV7(face_ds["test"], train=False)
BATCH = 64
face_train_loader = DataLoader(face_train_ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
face_test_loader  = DataLoader(face_test_ds, batch_size=128, shuffle=False, num_workers=2)

backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
face_model = nn.Sequential(backbone, nn.Dropout(0.4), nn.Linear(512, len(label_names_face))).to(device)
ema_model = copy.deepcopy(face_model)
for p in ema_model.parameters(): p.requires_grad = False
EMA_DECAY = 0.998

def update_ema():
    # FIX: iterate over the full state_dict (parameters AND buffers), not just .parameters().
    # BatchNorm's running_mean/running_var live in buffers — skipping them was the v6 bug.
    with torch.no_grad():
        msd = face_model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(EMA_DECAY).add_(model_v, alpha=1 - EMA_DECAY)
            else:
                ema_v.copy_(model_v)  # e.g. num_batches_tracked — just copy, don't average ints

face_crit = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
best_acc = 0.0

def mixup_batch(x, y, n, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1-lam) * x[perm]
    y1h = F.one_hot(y, n).float()
    return x_m, lam*y1h + (1-lam)*y1h[perm]

def cutmix_batch(x, y, n, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2], x.shape[3]
    cut_h, cut_w = int(H*np.sqrt(1-lam)), int(W*np.sqrt(1-lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy-cut_h//2,0,H), np.clip(cy+cut_h//2,0,H)
    x1, x2 = np.clip(cx-cut_w//2,0,W), np.clip(cx+cut_w//2,0,W)
    x_c = x.clone()
    x_c[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam_adj = 1 - ((y2-y1)*(x2-x1) / (H*W))
    y1h = F.one_hot(y, n).float()
    return x_c, lam_adj*y1h + (1-lam_adj)*y1h[perm]

def soft_ce(logits, soft_targets, weight=None):
    logp = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * logp).sum(dim=1)
    if weight is not None:
        loss = loss * (soft_targets * weight.unsqueeze(0)).sum(dim=1)
    return loss.mean()

@torch.no_grad()
def run_eval(model, ten_crop=True):
    model.eval()
    t_correct, t_total = 0, 0
    for x, y in face_test_loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            if ten_crop:
                views = [x, torch.flip(x, dims=[3])]
                h, w = x.shape[2], x.shape[3]
                ch, cw = int(h*0.85), int(w*0.85)
                corners = [(0,0), (0,w-cw), (h-ch,0), (h-ch,w-cw)]
                for (top, left) in corners:
                    crop = x[:, :, top:top+ch, left:left+cw]
                    crop = F.interpolate(crop, size=(h,w), mode='bilinear', align_corners=False)
                    views.append(crop)
                    views.append(torch.flip(crop, dims=[3]))
                out = sum(model(v).softmax(1) for v in views)
            else:
                out = model(x)
        t_correct += (out.argmax(1) == y).sum().item(); t_total += x.size(0)
    return t_correct / t_total

def train_epochs(n_epochs, optimizer, scheduler, phase_name, use_aug=True, use_ema=False):
    global best_acc
    for epoch in range(n_epochs):
        face_model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in face_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                r = np.random.rand()
                if use_aug and r < 0.35:
                    x_a, y_a = mixup_batch(x, y, len(label_names_face))
                    out = face_model(x_a); loss = soft_ce(out, y_a, class_weights_t)
                elif use_aug and r < 0.65:
                    x_a, y_a = cutmix_batch(x, y, len(label_names_face))
                    out = face_model(x_a); loss = soft_ce(out, y_a, class_weights_t)
                else:
                    out = face_model(x); loss = face_crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            if scheduler is not None: scheduler.step()
            if use_ema: update_ema()
            running_loss += loss.item() * x.size(0)
            with torch.no_grad():
                preds = face_model(x).argmax(1)
            correct += (preds == y).sum().item(); total += x.size(0)

        test_acc = run_eval(face_model)
        ema_acc = run_eval(ema_model) if use_ema else 0.0
        better_model, better_acc = (ema_model, ema_acc) if ema_acc > test_acc else (face_model, test_acc)
        if better_acc > best_acc:
            best_acc = better_acc
            torch.save({"state_dict": better_model.state_dict(), "labels": label_names_face, "test_acc": best_acc},
                       SAVE_PATH)
        print(f"[{phase_name}] epoch {epoch+1}/{n_epochs} loss={running_loss/total:.4f} "
              f"train_acc={correct/total:.4f} test_acc={test_acc:.4f} ema_acc={ema_acc:.4f} best={best_acc:.4f}")

print("\n-- Phase 1: frozen backbone, head only --")
for p in backbone.parameters(): p.requires_grad = False
phase1_opt = torch.optim.AdamW([p for p in face_model.parameters() if p.requires_grad], lr=1e-3)
phase1_sched = torch.optim.lr_scheduler.OneCycleLR(phase1_opt, max_lr=1e-3, steps_per_epoch=len(face_train_loader), epochs=4)
train_epochs(4, phase1_opt, phase1_sched, "phase1-head", use_aug=False)

print("\n-- Phase 2: unfreeze last blocks --")
for name, p in backbone.named_parameters():
    if any(k in name for k in ["block8","repeat_3","last_linear","last_bn"]):
        p.requires_grad = True
phase2_opt = torch.optim.AdamW([p for p in face_model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
phase2_sched = torch.optim.lr_scheduler.OneCycleLR(phase2_opt, max_lr=3e-4, steps_per_epoch=len(face_train_loader), epochs=8)
train_epochs(8, phase2_opt, phase2_sched, "phase2-partial-finetune")

print("\n-- Phase 3: full fine-tune with EMA --")
for p in face_model.parameters(): p.requires_grad = True
ema_model.load_state_dict(face_model.state_dict())  # sync EMA (weights + buffers) to current state
phase3_opt = torch.optim.AdamW(face_model.parameters(), lr=5e-5, weight_decay=1e-4)
phase3_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(phase3_opt, T_0=len(face_train_loader)*3, T_mult=1)
train_epochs(18, phase3_opt, phase3_sched, "phase3-full-finetune", use_ema=True)

print(f"\nFACE v7 best test accuracy: {best_acc:.4f} (saved to {SAVE_PATH})")
