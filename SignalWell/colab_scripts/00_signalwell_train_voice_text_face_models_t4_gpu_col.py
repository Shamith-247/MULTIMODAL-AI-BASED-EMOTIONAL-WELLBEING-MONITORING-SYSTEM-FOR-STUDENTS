# ============================================================
#  SignalWell — Train Voice / Text / Face models (T4 GPU, Colab)
#  Runtime -> Change runtime type -> T4 GPU, then Run all.
#  Total runtime on a T4: roughly 25–40 min for all three.
# ============================================================

!pip -q install "transformers>=4.42" "datasets>=2.19" accelerate evaluate librosa soundfile scikit-learn --upgrade

import os, re, io, zipfile, requests, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                           Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, DataCollatorWithPadding)
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
os.makedirs("/content/outputs", exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# ============================================================
# PART 1 — TEXT: fine-tune DistilBERT on dair-ai/emotion (6 classes)
# ============================================================
print("\n=== [1/3] TEXT — fine-tuning DistilBERT on dair-ai/emotion ===")

text_ds = load_dataset("dair-ai/emotion", "split")
label_names_text = text_ds["train"].features["label"].names   # sadness, joy, love, anger, fear, surprise
print("Text labels:", label_names_text)

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tok_fn(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=64)
text_ds_tok = text_ds.map(tok_fn, batched=True)
text_ds_tok = text_ds_tok.remove_columns(["text"]).rename_column("label", "labels")
text_ds_tok.set_format("torch")

text_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(label_names_text),
    id2label={i:n for i,n in enumerate(label_names_text)},
    label2id={n:i for i,n in enumerate(label_names_text)}
).to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

text_args = TrainingArguments(
    output_dir="/content/outputs/text_ckpt",
    eval_strategy="epoch", save_strategy="no",
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    num_train_epochs=3, learning_rate=2e-5, weight_decay=0.01,
    logging_steps=100, report_to=[], fp16=torch.cuda.is_available()
)
text_trainer = Trainer(
    model=text_model, args=text_args,
    train_dataset=text_ds_tok["train"], eval_dataset=text_ds_tok["validation"],
    compute_metrics=compute_metrics
)
text_trainer.train()
text_test_metrics = text_trainer.evaluate(text_ds_tok["test"])
print("TEXT test metrics:", text_test_metrics)

text_model.save_pretrained("/content/outputs/text_model")
tok.save_pretrained("/content/outputs/text_model")


# ============================================================
# PART 2 — FACE: small CNN trained from scratch on FER2013 (7 classes)
# ============================================================
print("\n=== [2/3] FACE — training CNN on FER2013 (mehmet-3emin/fer2013-cleaned) ===")

face_ds = load_dataset("mehmet-3emin/fer2013-cleaned")
label_names_face = face_ds["train"].features["label"].names
print("Face labels:", label_names_face)

class FaceTorchDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"].convert("L").resize((48,48))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.tensor(arr).unsqueeze(0)  # 1x48x48
        return tensor, item["label"]

face_train_ds = FaceTorchDataset(face_ds["train"])
face_test_key = "test" if "test" in face_ds else "validation"
face_test_ds  = FaceTorchDataset(face_ds[face_test_key])

face_train_loader = DataLoader(face_train_ds, batch_size=128, shuffle=True, num_workers=2)
face_test_loader  = DataLoader(face_test_ds, batch_size=256, shuffle=False, num_workers=2)

class FaceCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 24x24
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 12x12
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),# 6x6
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


# ============================================================
# PART 3 — VOICE: fine-tune Wav2Vec2 on RAVDESS (8 emotion classes)
# ============================================================
print("\n=== [3/3] VOICE — fine-tuning Wav2Vec2 on RAVDESS ===")

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ravdess_dir = "/content/ravdess"
if not os.path.exists(ravdess_dir):
    os.makedirs(ravdess_dir, exist_ok=True)
    print("Downloading RAVDESS speech set (~215MB)...")
    r = requests.get(RAVDESS_URL, timeout=180)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(ravdess_dir)

EMOTION_MAP = {"01":"neutral","02":"calm","03":"happy","04":"sad",
               "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
label_names_voice = list(EMOTION_MAP.values())
label2id_voice = {n:i for i,n in enumerate(label_names_voice)}

wav_paths, wav_labels = [], []
for root, _, files in os.walk(ravdess_dir):
    for f in files:
        if f.endswith(".wav"):
            code = f.split("-")[2]
            wav_paths.append(os.path.join(root, f))
            wav_labels.append(label2id_voice[EMOTION_MAP[code]])
print(f"Found {len(wav_paths)} RAVDESS clips")

idx = list(range(len(wav_paths))); random.shuffle(idx)
split = int(0.85 * len(idx))
train_idx, test_idx = idx[:split], idx[split:]

feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", do_normalize=True)
MAX_LEN = 16000 * 4  # 4 seconds

class RavdessDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx_ = self.indices[i]
        import librosa
        wav, _ = librosa.load(wav_paths[idx_], sr=16000, mono=True)
        wav = wav[:MAX_LEN]
        feats = feat_extractor(wav, sampling_rate=16000, max_length=MAX_LEN,
                                padding="max_length", truncation=True, return_tensors="pt")
        return {"input_values": feats["input_values"][0], "labels": wav_labels[idx_]}

voice_train_ds = RavdessDataset(train_idx)
voice_test_ds  = RavdessDataset(test_idx)

voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=len(label_names_voice),
    id2label={i:n for i,n in enumerate(label_names_voice)},
    label2id=label2id_voice
).to(device)
voice_model.freeze_feature_encoder()  # keep the CNN front-end frozen — faster + avoids overfitting on 1.4k clips

def voice_collate(batch):
    input_values = torch.stack([b["input_values"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch])
    return {"input_values": input_values, "labels": labels}

voice_args = TrainingArguments(
    output_dir="/content/outputs/voice_ckpt",
    eval_strategy="epoch", save_strategy="no",
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=8, learning_rate=3e-5, warmup_ratio=0.1,
    logging_steps=20, report_to=[], fp16=torch.cuda.is_available()
)

def voice_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]}

voice_trainer = Trainer(
    model=voice_model, args=voice_args,
    train_dataset=voice_train_ds, eval_dataset=voice_test_ds,
    data_collator=voice_collate, compute_metrics=voice_compute_metrics
)
voice_trainer.train()
voice_test_metrics = voice_trainer.evaluate()
print("VOICE test metrics:", voice_test_metrics)

voice_model.save_pretrained("/content/outputs/voice_model")
feat_extractor.save_pretrained("/content/outputs/voice_model")


# ============================================================
# SUMMARY
# ============================================================
print("\n================ DONE ================")
print("Text  test:", text_test_metrics)
print("Face  test accuracy:", correct/total)
print("Voice test:", voice_test_metrics)
print("Saved to: /content/outputs/{text_model, face_model.pt, voice_model}")
