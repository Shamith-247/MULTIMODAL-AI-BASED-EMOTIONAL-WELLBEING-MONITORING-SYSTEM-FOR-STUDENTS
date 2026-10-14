# ============================================================
# PART 2 (v3) — FACE: ResNet18 transfer learning (frozen -> fine-tune)
# ============================================================
print("\n=== [2/3] FACE v3 — ResNet18 transfer learning ===")

import torchvision.models as models
import torchvision.transforms as T

if "face_ds" not in dir():
    face_ds = load_dataset("abhilash88/fer2013-enhanced")
label_names_face = face_ds["train"].features["emotion"].names
print("Face labels:", label_names_face)

train_labels_list = face_ds["train"]["emotion"]
class_counts = np.bincount(train_labels_list, minlength=len(label_names_face))
class_weights = (1.0 / np.maximum(class_counts, 1))
class_weights = class_weights / class_weights.sum() * len(label_names_face)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

IMG_SIZE = 160  # upscaled from 48x48 so pretrained ResNet features are meaningful, kept modest for T4 speed
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class FaceDatasetV3(Dataset):
    def __init__(self, hf_split, train=False):
        self.data = hf_split
        self.train = train
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        arr = np.array(item["image"], dtype=np.uint8)
        arr3 = np.stack([arr, arr, arr], axis=-1)  # grayscale -> fake RGB, 48x48x3
        tf = train_tf if self.train else eval_tf
        tensor = tf(arr3)
        return tensor, item["emotion"]

face_train_ds = FaceDatasetV3(face_ds["train"], train=True)
face_test_ds  = FaceDatasetV3(face_ds["test"], train=False)

BATCH = 64
face_train_loader = DataLoader(face_train_ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
face_test_loader  = DataLoader(face_test_ds, batch_size=128, shuffle=False, num_workers=2)

face_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
face_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(face_model.fc.in_features, len(label_names_face))
)
face_model = face_model.to(device)

face_crit = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
best_acc = 0.0

def run_eval():
    face_model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y in face_test_loader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = face_model(x)
            t_correct += (out.argmax(1) == y).sum().item(); t_total += x.size(0)
    return t_correct / t_total

def train_epochs(n_epochs, optimizer, scheduler, phase_name):
    global best_acc
    for epoch in range(n_epochs):
        face_model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in face_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = face_model(x)
                loss = face_crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None: scheduler.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item(); total += x.size(0)
        test_acc = run_eval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"state_dict": face_model.state_dict(), "labels": label_names_face, "test_acc": best_acc},
                       "/content/outputs/face_model_resnet18_best.pt")
        print(f"[{phase_name}] epoch {epoch+1}/{n_epochs} loss={running_loss/total:.4f} "
              f"train_acc={correct/total:.4f} test_acc={test_acc:.4f} best={best_acc:.4f}")

# ---- Phase 1: freeze backbone, train only the new classifier head (fast warmup) ----
print("\n-- Phase 1: frozen backbone, head only --")
for name, p in face_model.named_parameters():
    p.requires_grad = "fc" in name
phase1_opt = torch.optim.AdamW([p for p in face_model.parameters() if p.requires_grad], lr=1e-3)
phase1_sched = torch.optim.lr_scheduler.OneCycleLR(
    phase1_opt, max_lr=1e-3, steps_per_epoch=len(face_train_loader), epochs=4
)
train_epochs(4, phase1_opt, phase1_sched, "phase1-head")

# ---- Phase 2: unfreeze everything, fine-tune the whole network at a low LR ----
print("\n-- Phase 2: full fine-tune --")
for p in face_model.parameters():
    p.requires_grad = True
phase2_opt = torch.optim.AdamW(face_model.parameters(), lr=1e-4, weight_decay=1e-4)
phase2_sched = torch.optim.lr_scheduler.OneCycleLR(
    phase2_opt, max_lr=1e-4, steps_per_epoch=len(face_train_loader), epochs=14
)
train_epochs(14, phase2_opt, phase2_sched, "phase2-finetune")

print(f"\nFACE v3 (ResNet18) best test accuracy: {best_acc:.4f} (saved to /content/outputs/face_model_resnet18_best.pt)")
