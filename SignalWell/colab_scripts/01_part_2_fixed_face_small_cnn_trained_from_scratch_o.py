# ============================================================
# PART 2 (fixed) — FACE: small CNN trained from scratch on FER2013
# ============================================================
print("\n=== [2/3] FACE — training CNN on FER2013 (abhilash88/fer2013-enhanced) ===")

face_ds = load_dataset("abhilash88/fer2013-enhanced")
label_names_face = face_ds["train"].features["emotion"].names
print("Face labels:", label_names_face)

class FaceTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        arr = np.array(item["image"], dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.tensor(arr).unsqueeze(0)  # 1x48x48
        return tensor, item["emotion"]

face_train_ds = FaceTorchDataset(face_ds["train"])
face_test_ds  = FaceTorchDataset(face_ds["test"])

face_train_loader = DataLoader(face_train_ds, batch_size=128, shuffle=True, num_workers=2)
face_test_loader  = DataLoader(face_test_ds, batch_size=256, shuffle=False, num_workers=2)

class FaceCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(128*6*6, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

face_model = FaceCNN(len(label_names_face)).to(device)
face_opt = torch.optim.AdamW(face_model.parameters(), lr=1e-3, weight_decay=1e-4)
face_sched = torch.optim.lr_scheduler.CosineAnnealingLR(face_opt, T_max=12)
face_crit = nn.CrossEntropyLoss()

FACE_EPOCHS = 12
for epoch in range(FACE_EPOCHS):
    face_model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in face_train_loader:
        x, y = x.to(device), y.to(device)
        face_opt.zero_grad()
        out = face_model(x)
        loss = face_crit(out, y)
        loss.backward(); face_opt.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)
    face_sched.step()
    print(f"[face] epoch {epoch+1}/{FACE_EPOCHS} loss={running_loss/total:.4f} train_acc={correct/total:.4f}")

face_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in face_test_loader:
        x, y = x.to(device), y.to(device)
        out = face_model(x)
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)
print(f"FACE test accuracy: {correct/total:.4f}")

torch.save({"state_dict": face_model.state_dict(), "labels": label_names_face},
           "/content/outputs/face_model.pt")
