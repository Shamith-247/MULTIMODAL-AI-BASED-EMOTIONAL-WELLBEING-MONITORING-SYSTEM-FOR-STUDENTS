# ============================================================
# PART 2 (v2) — FACE: deeper CNN + augmentation + class weighting
# ============================================================
print("\n=== [2/3] FACE v2 — improved training ===")

from PIL import Image
import torchvision.transforms as T

if "face_ds" not in dir():
    face_ds = load_dataset("abhilash88/fer2013-enhanced")
label_names_face = face_ds["train"].features["emotion"].names
print("Face labels:", label_names_face)

# ---- class weights (inverse frequency) to counter the disgust/happy imbalance ----
train_labels_list = face_ds["train"]["emotion"]
class_counts = np.bincount(train_labels_list, minlength=len(label_names_face))
class_weights = (1.0 / np.maximum(class_counts, 1))
class_weights = class_weights / class_weights.sum() * len(label_names_face)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class counts:", dict(zip(label_names_face, class_counts.tolist())))
print("Class weights:", dict(zip(label_names_face, np.round(class_weights, 2).tolist())))

# ---- augmentation for train, plain normalize for eval ----
train_tf = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(12),
    T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
    T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
])

class FaceTorchDatasetV2(Dataset):
    def __init__(self, hf_split, train=False):
        self.data = hf_split
        self.train = train
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        arr = np.array(item["image"], dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        tensor = torch.tensor(arr).unsqueeze(0)  # 1x48x48, values in [0,1]
        if self.train:
            tensor = train_tf(tensor)
        tensor = (tensor - 0.5) / 0.5
        return tensor, item["emotion"]

face_train_ds = FaceTorchDatasetV2(face_ds["train"], train=True)
face_test_ds  = FaceTorchDatasetV2(face_ds["test"], train=False)

face_train_loader = DataLoader(face_train_ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
face_test_loader  = DataLoader(face_test_ds, batch_size=256, shuffle=False, num_workers=2)

# ---- deeper CNN with an extra block + global average pooling ----
class FaceCNNv2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2)
            )
        self.b1 = block(1, 64)     # 48->24
        self.b2 = block(64, 128)   # 24->12
        self.b3 = block(128, 256)  # 12->6
        self.b4 = block(256, 256)  # 6->3
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = self.gap(x)
        return self.classifier(x)

face_model = FaceCNNv2(len(label_names_face)).to(device)
face_opt = torch.optim.AdamW(face_model.parameters(), lr=1.5e-3, weight_decay=5e-4)
FACE_EPOCHS = 30
face_sched = torch.optim.lr_scheduler.OneCycleLR(
    face_opt, max_lr=1.5e-3, steps_per_epoch=len(face_train_loader), epochs=FACE_EPOCHS
)
face_crit = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

best_acc = 0.0
for epoch in range(FACE_EPOCHS):
    face_model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in face_train_loader:
        x, y = x.to(device), y.to(device)
        face_opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = face_model(x)
            loss = face_crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(face_opt)
        scaler.update()
        face_sched.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)

    face_model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y in face_test_loader:
            x, y = x.to(device), y.to(device)
            out = face_model(x)
            t_correct += (out.argmax(1) == y).sum().item(); t_total += x.size(0)
    test_acc = t_correct / t_total
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({"state_dict": face_model.state_dict(), "labels": label_names_face, "test_acc": best_acc},
                   "/content/outputs/face_model_best.pt")

    print(f"[face v2] epoch {epoch+1}/{FACE_EPOCHS} "
          f"loss={running_loss/total:.4f} train_acc={correct/total:.4f} "
          f"test_acc={test_acc:.4f} best={best_acc:.4f}")

print(f"\nFACE v2 best test accuracy: {best_acc:.4f}  (saved to /content/outputs/face_model_best.pt)")
