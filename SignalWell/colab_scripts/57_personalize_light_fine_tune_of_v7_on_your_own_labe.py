# ============================================================
# PERSONALIZE — light fine-tune of v7 on your own labeled photos
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy

class PersonalDataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples = samples
        self.train = train
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        with torch.no_grad():
            face_tensor = mtcnn(img)  # reuse the training MTCNN instance
        if face_tensor is None:
            # fallback: just resize the whole image if no face detected
            face_pil = img.resize((160,160))
        else:
            face_raw = face_tensor.clamp(0, 255).byte()
            face_pil = T.ToPILImage()(face_raw)
        face_pil = apply_clahe(face_pil)  # same lighting correction as inference
        tf = train_aug if self.train else eval_tf
        return tf(face_pil), LABELS_ORDER.index(label)

# Small train/val split of your own photos
random.shuffle(my_dataset)
split_point = max(1, int(len(my_dataset) * 0.85))
my_train = my_dataset[:split_point]
my_val = my_dataset[split_point:] if len(my_dataset) > split_point else my_dataset[-2:]

personal_train_ds = PersonalDataset(my_train, train=True)
personal_val_ds = PersonalDataset(my_val, train=False)
personal_train_loader = DataLoader(personal_train_ds, batch_size=4, shuffle=True)
personal_val_loader = DataLoader(personal_val_ds, batch_size=4, shuffle=False)

# Start from your best v7 checkpoint, keep a separate copy so the original stays intact
personal_model = copy.deepcopy(model)
personal_model.load_state_dict(ckpt["state_dict"])

# Only fine-tune the classifier + last backbone block — freeze everything else
# to avoid catastrophic forgetting on such a small dataset
for p in personal_model.parameters():
    p.requires_grad = False
for p in personal_model.fc.parameters():
    p.requires_grad = True
for name, p in personal_model.backbone.named_parameters():
    if any(k in name for k in ["block8", "last_linear", "last_bn"]):
        p.requires_grad = True

opt = torch.optim.AdamW([p for p in personal_model.parameters() if p.requires_grad],
                         lr=1e-4, weight_decay=1e-3)

PERSONAL_SAVE_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v7_personalized.pt"
best_personal_acc = 0.0
EPOCHS = 15

for epoch in range(EPOCHS):
    personal_model.train()
    total, correct = 0, 0
    for x, y in personal_train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = personal_model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)

    personal_model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for x, y in personal_val_loader:
            x, y = x.to(device), y.to(device)
            out = personal_model(x)
            v_correct += (out.argmax(1) == y).sum().item(); v_total += x.size(0)
    val_acc = v_correct / max(v_total, 1)
    train_acc = correct / total

    print(f"epoch {epoch+1}/{EPOCHS} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
    if val_acc >= best_personal_acc:
        best_personal_acc = val_acc
        torch.save({"state_dict": personal_model.state_dict(), "labels": LABELS_ORDER},
                   PERSONAL_SAVE_PATH)

print(f"\nBest personalized val accuracy: {best_personal_acc:.3f}")
print(f"Saved to: {PERSONAL_SAVE_PATH}")
