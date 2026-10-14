# ============================================================
# VOICE SER v6 — WavLM-base-plus + attention pooling + speaker-independent split
# Single cell. MUST run on a completely FRESH runtime (Runtime > Disconnect and
# delete runtime > reconnect > confirm T4 GPU) as the very first thing you run.
# ============================================================
!pip uninstall -y numpy -q
!pip install "numpy==2.1.3" --force-reinstall --no-cache-dir -q
!pip install -q -U "transformers>=4.46,<5.0" "accelerate>=0.34" librosa soundfile kaggle scikit-learn

import os, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              confusion_matrix, classification_report)
from google.colab import drive

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
FINAL_DIR = "/content/drive/MyDrive/signalwell-models/voice_model_v6_wavlm"
os.makedirs(FINAL_DIR, exist_ok=True)

# ---- >>> YOUR KAGGLE CREDENTIALS <<< ----
KAGGLE_USERNAME = "your_kaggle_username_here"
KAGGLE_KEY = "your_kaggle_api_key_here"
# ------------------------------------------
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# STEP 4 — dataset download + standardized 7-class label mapping
# ============================================================
DATA_DIR = "/content/speech_emotion"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d dmitrybabko/speech-emotion-recognition-en --unzip -p {DATA_DIR}")

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label2id = {n: i for i, n in enumerate(LABELS)}

# RAVDESS "calm" (code 02) is explicitly folded into "neutral" — stated openly, not silently.
RAVDESS_CODE = {"01":"neutral","02":"neutral","03":"happy","04":"sad",
                "05":"angry","06":"fear","07":"disgust","08":"surprise"}
CREMA_CODE = {"ANG":"angry","DIS":"disgust","FEA":"fear","HAP":"happy","NEU":"neutral","SAD":"sad"}

def label_from_path(path):
    p = path.replace("\\", "/"); lower = p.lower(); fname = os.path.basename(p)
    if "ravdess" in lower:
        parts = fname.replace(".wav", "").split("-")
        return RAVDESS_CODE.get(parts[2]) if len(parts) >= 3 else None
    if "crema" in lower:
        parts = fname.replace(".wav", "").split("_")
        return CREMA_CODE.get(parts[2]) if len(parts) >= 3 else None
    if "savee" in lower:
        stem = fname.replace(".wav", "").lower()
        if "_" in stem:
            tail = stem.split("_")[-1]
        else:
            tail = stem
        if tail.startswith("sa"): return "sad"
        if tail.startswith("su"): return "surprise"
        if tail.startswith("a"): return "angry"
        if tail.startswith("d"): return "disgust"
        if tail.startswith("f"): return "fear"
        if tail.startswith("h"): return "happy"
        if tail.startswith("n"): return "neutral"
        return None
    if "tess" in lower:
        if "disgust" in lower: return "disgust"
        if "fear" in lower: return "fear"
        if "happy" in lower: return "happy"
        if "neutral" in lower: return "neutral"
        if "sad" in lower: return "sad"
        if "angry" in lower or "anger" in lower: return "angry"
        if "ps" in lower or "surprise" in lower or "pleasant" in lower: return "surprise"
        return None
    return None

# ---- Speaker ID extraction, per dataset, needed for speaker-independent splitting (STEP 3) ----
def dataset_name_from_path(path):
    lower = path.lower()
    if "ravdess" in lower: return "ravdess"
    if "crema" in lower: return "crema"
    if "savee" in lower: return "savee"
    if "tess" in lower: return "tess"
    return "unknown"

def speaker_from_path(path):
    p = path.replace("\\", "/"); lower = p.lower(); fname = os.path.basename(p)
    ds = dataset_name_from_path(path)
    if ds == "ravdess":
        parts = fname.replace(".wav", "").split("-")
        return f"ravdess_actor{parts[6]}" if len(parts) >= 7 else "ravdess_unknown"
    if ds == "crema":
        parts = fname.replace(".wav", "").split("_")
        return f"crema_{parts[0]}" if len(parts) >= 1 else "crema_unknown"
    if ds == "savee":
        # SAVEE speaker codes are folder names: DC, JE, JK, KL
        parts = p.split("/")
        for part in parts:
            if part.upper() in ("DC", "JE", "JK", "KL"):
                return f"savee_{part.upper()}"
        return "savee_unknown"
    if ds == "tess":
        if "oaf" in lower: return "tess_OAF"
        if "yaf" in lower: return "tess_YAF"
        return "tess_unknown"
    return "unknown_unknown"

wav_paths, wav_labels, wav_speakers, wav_datasets = [], [], [], []
for root, _, files_ in os.walk(DATA_DIR):
    for f in files_:
        if f.lower().endswith(".wav"):
            full = os.path.join(root, f)
            lab = label_from_path(full)
            if lab is not None:
                wav_paths.append(full)
                wav_labels.append(label2id[lab])
                wav_speakers.append(speaker_from_path(full))
                wav_datasets.append(dataset_name_from_path(full))

wav_labels = np.array(wav_labels)
wav_speakers = np.array(wav_speakers)
wav_datasets = np.array(wav_datasets)

print(f"\nTotal usable clips: {len(wav_paths)}")
print(f"Unique speakers/groups: {len(set(wav_speakers))}")
for ds in sorted(set(wav_datasets)):
    mask = wav_datasets == ds
    print(f"  {ds}: {mask.sum()} samples, {len(set(wav_speakers[mask]))} speaker-groups, "
          f"emotions present: {sorted(set(LABELS[l] for l in wav_labels[mask]))}")

print("\nOverall class distribution:")
overall_counts = np.bincount(wav_labels, minlength=len(LABELS))
for name, c in zip(LABELS, overall_counts):
    print(f"  {name}: {c}")

# ============================================================
# STEP 3 — speaker-independent 70/15/15 split (grouped by speaker, done globally)
# NOTE / LIMITATION: TESS has only 2 total speakers and SAVEE only 4. A speaker-independent
# split at the level of an INDIVIDUAL dataset isn't meaningful for those two (you cannot put
# 70/15/15 of 2 speakers). Instead we group-split GLOBALLY across all ~121 speaker-groups
# combined (24 RAVDESS + ~91 CREMA-D + 2 TESS + 4 SAVEE). This guarantees no speaker's clips
# ever appear in more than one split, at the cost that a given split may end up with zero or
# very few samples from TESS/SAVEE specifically if their few speakers land elsewhere by chance.
# We print the resulting per-dataset counts below so this is visible, not hidden.
# ============================================================
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(gss1.split(np.zeros(len(wav_paths)), wav_labels, groups=wav_speakers))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=SEED)
train_idx_rel, val_idx_rel = next(gss2.split(np.zeros(len(trainval_idx)), wav_labels[trainval_idx],
                                              groups=wav_speakers[trainval_idx]))
train_idx = trainval_idx[train_idx_rel]
val_idx = trainval_idx[val_idx_rel]

# Verify zero speaker overlap (hard assertion, not just a claim)
train_speakers = set(wav_speakers[train_idx])
val_speakers = set(wav_speakers[val_idx])
test_speakers = set(wav_speakers[test_idx])
assert not (train_speakers & val_speakers), "LEAK: speaker in both train and val"
assert not (train_speakers & test_speakers), "LEAK: speaker in both train and test"
assert not (val_speakers & test_speakers), "LEAK: speaker in both val and test"
print("\nSpeaker-independence verified: zero overlap between train/val/test speaker sets.")

print(f"\nSplit sizes: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    print(f"\n{split_name} per-dataset breakdown:")
    for ds in sorted(set(wav_datasets)):
        mask = wav_datasets[split_idx] == ds
        print(f"  {ds}: {mask.sum()} samples")
    print(f"{split_name} class distribution:", dict(zip(LABELS, np.bincount(wav_labels[split_idx], minlength=len(LABELS)).tolist())))

# ============================================================
# STEP 7 — class weights from the TRAIN split only (not global, avoids val/test leakage into weighting)
# ============================================================
train_counts = np.bincount(wav_labels[train_idx], minlength=len(LABELS))
class_weights = (1.0 / np.maximum(train_counts, 1))
class_weights = class_weights / class_weights.sum() * len(LABELS)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("\nClass weights (train-derived):", dict(zip(LABELS, np.round(class_weights, 3).tolist())))

# ============================================================
# STEP 5/6 — preprocessing + augmentation (train-only augmentation)
# ============================================================
MODEL_NAME = "microsoft/wavlm-base-plus"
feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, do_normalize=True)
MAX_LEN = 16000 * 4  # 16kHz, 4s max — covers the vast majority of clips in all 4 datasets

def augment_waveform(wav):
    import librosa
    if random.random() < 0.5:
        wav = wav + np.random.normal(0, 0.005, size=wav.shape).astype(np.float32)
    if random.random() < 0.5:
        wav = wav * random.uniform(0.85, 1.15)
    if random.random() < 0.3:
        shift = int(len(wav) * random.uniform(-0.1, 0.1))
        wav = np.roll(wav, shift)
    if random.random() < 0.3:
        n_steps = random.uniform(-2, 2)
        wav = librosa.effects.pitch_shift(wav, sr=16000, n_steps=n_steps)
    if random.random() < 0.3:
        rate = random.uniform(0.9, 1.1)
        wav = librosa.effects.time_stretch(wav, rate=rate)
    return wav.astype(np.float32)

class VoiceDataset(Dataset):
    def __init__(self, indices, train=False):
        self.indices = indices
        self.train = train
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx_ = self.indices[i]
        import librosa
        wav, _ = librosa.load(wav_paths[idx_], sr=16000, mono=True)
        if self.train:
            wav = augment_waveform(wav)
        wav = wav[:MAX_LEN]
        feats = feat_extractor(wav, sampling_rate=16000, max_length=MAX_LEN,
                                padding="max_length", truncation=True, return_tensors="pt")
        return {"input_values": feats["input_values"][0],
                "attention_mask": feats["attention_mask"][0],
                "labels": int(wav_labels[idx_])}

def collate_fn(batch):
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch]),
    }

train_ds = VoiceDataset(train_idx, train=True)
val_ds = VoiceDataset(val_idx, train=False)
test_ds = VoiceDataset(test_idx, train=False)

# ============================================================
# STEP 8/9 — WavLM-base-plus + attention pooling (instead of plain mean pooling)
# ============================================================
class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states, attention_mask=None):
        scores = self.attn(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)

class WavLMEmotionClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels, dropout=0.3):
        super().__init__()
        self.encoder = WavLMModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.pool = AttentionPool(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        feat_mask = None
        if attention_mask is not None:
            feat_mask = self.encoder._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        pooled = self.pool(hidden_states, feat_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

model = WavLMEmotionClassifier(MODEL_NAME, len(LABELS)).to(device)

# ============================================================
# STEP 10 — training loop: AdamW, differential LR (encoder << head), gradient clipping,
# mixed precision, early stopping on val macro F1, checkpoint = best val macro F1 (not last epoch)
# ============================================================
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(input_values, attention_mask)
            preds = logits.argmax(-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return acc, f1_macro, f1_weighted, all_preds, all_labels

def train_phase(model, train_loader, val_loader, encoder_lr, head_lr, n_epochs, patience, phase_name):
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.pool.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder_params if p.requires_grad], "lr": encoder_lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": head_lr, "weight_decay": 1e-3},
    ])
    total_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[encoder_lr, head_lr], total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss, total = 0.0, 0
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(input_values, attention_mask)
                loss = F.cross_entropy(logits, labels, weight=class_weights_t, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate_model(model, val_loader)
        print(f"[{phase_name}] epoch {epoch+1}/{n_epochs} "
              f"train_loss={running_loss/total:.4f} val_acc={val_acc:.4f} "
              f"val_f1_macro={val_f1_macro:.4f} val_f1_weighted={val_f1_weighted:.4f}")

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(FINAL_DIR, "best_model.pt"))
            print(f"  -> new best (val_f1_macro={best_f1:.4f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  -> early stopping (no improvement for {patience} epochs)")
                break

    return best_f1

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)

# ---- Phase A: encoder frozen, only pooling+classifier train ----
print("\n== Phase A: frozen WavLM encoder, train pooling+classifier only ==")
for p in model.encoder.parameters():
    p.requires_grad = False
train_phase(model, train_loader, val_loader, encoder_lr=0.0, head_lr=1e-4, n_epochs=8, patience=3, phase_name="phaseA-frozen")

# ---- Phase B: unfreeze encoder, fine-tune with a much smaller LR than the head ----
print("\n== Phase B: unfrozen WavLM encoder, differential LR fine-tune ==")
model.load_state_dict(torch.load(os.path.join(FINAL_DIR, "best_model.pt")))  # resume from phase A's best
for p in model.encoder.parameters():
    p.requires_grad = True
final_best_f1 = train_phase(model, train_loader, val_loader, encoder_lr=1e-5, head_lr=5e-5,
                             n_epochs=10, patience=4, phase_name="phaseB-finetune")

# ============================================================
# STEP 14 — ONE final evaluation on held-out test set, using the best checkpoint
# ============================================================
print("\n================ FINAL TEST EVALUATION (run once) ================")
model.load_state_dict(torch.load(os.path.join(FINAL_DIR, "best_model.pt")))
test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = evaluate_model(model, test_loader)
precision = precision_score(test_labels, test_preds, average="macro", zero_division=0)
recall = recall_score(test_labels, test_preds, average="macro", zero_division=0)

print(f"Test Accuracy:      {test_acc:.4f}")
print(f"Test Macro F1:      {test_f1_macro:.4f}")
print(f"Test Weighted F1:   {test_f1_weighted:.4f}")
print(f"Test Macro Precision: {precision:.4f}")
print(f"Test Macro Recall:    {recall:.4f}")

print("\nPer-class report:")
print(classification_report(test_labels, test_preds, target_names=LABELS, zero_division=0))

print("Confusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(LABELS))))
print("        " + " ".join(f"{n[:4]:>6}" for n in LABELS))
for i, row in enumerate(cm):
    print(f"{LABELS[i][:6]:>6}  " + " ".join(f"{v:>6}" for v in row))

# Save config alongside the checkpoint for reproducibility
config = {
    "base_model": MODEL_NAME, "labels": LABELS, "seed": SEED,
    "test_accuracy": test_acc, "test_f1_macro": test_f1_macro,
    "test_f1_weighted": test_f1_weighted, "test_precision_macro": precision,
    "test_recall_macro": recall,
}
with open(os.path.join(FINAL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nFinal model + config saved to: {FINAL_DIR}")
