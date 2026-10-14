# ============================================================
# PART 3 (v5) — VOICE: two-phase (frozen->unfrozen) + pitch shift
# + multi-window TTA. Single cell — run on a FRESH runtime.
# ============================================================
!pip install -q -U "numpy>=2.0" "transformers>=4.46,<5.0" "datasets>=3.0" "accelerate>=0.34" "evaluate>=0.4.3" "huggingface_hub>=0.25" librosa soundfile kaggle

import numpy as np
print("numpy:", np.__version__)
import transformers, datasets, evaluate as evaluate_lib, accelerate
print("transformers:", transformers.__version__, "| datasets:", datasets.__version__)

import os, json, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
                           TrainingArguments, Trainer)
import evaluate
from google.colab import drive

drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
DRIVE_CKPT = "/content/drive/MyDrive/signalwell-models/voice_ckpt_v5"
FINAL_DIR = "/content/drive/MyDrive/signalwell-models/voice_model_v5"

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

DATA_DIR = "/content/speech_emotion"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d dmitrybabko/speech-emotion-recognition-en --unzip -p {DATA_DIR}")

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label2id = {n: i for i, n in enumerate(LABELS)}

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
        tail = stem.split("_")[-1] if "_" in stem else stem
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

wav_paths, wav_labels = [], []
for root, _, files_ in os.walk(DATA_DIR):
    for f in files_:
        if f.lower().endswith(".wav"):
            full = os.path.join(root, f)
            lab = label_from_path(full)
            if lab is not None:
                wav_paths.append(full); wav_labels.append(label2id[lab])

print(f"Total usable clips: {len(wav_paths)}")
counts = np.bincount(wav_labels, minlength=len(LABELS))
print("Class distribution:", dict(zip(LABELS, counts.tolist())))

class_weights = (1.0 / np.maximum(counts, 1))
class_weights = class_weights / class_weights.sum() * len(LABELS)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

idx = list(range(len(wav_paths))); random.seed(42); random.shuffle(idx)
split = int(0.85 * len(idx))
train_idx, test_idx = idx[:split], idx[split:]

feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", do_normalize=True)
MAX_LEN = 16000 * 4

def augment_waveform(wav):
    import librosa
    if random.random() < 0.5:
        wav = wav + np.random.normal(0, 0.005, size=wav.shape).astype(np.float32)
    if random.random() < 0.5:
        wav = wav * random.uniform(0.85, 1.15)
    if random.random() < 0.3:
        shift = int(len(wav) * random.uniform(-0.1, 0.1))
        wav = np.roll(wav, shift)
    if random.random() < 0.4:
        n_steps = random.uniform(-2, 2)
        wav = librosa.effects.pitch_shift(wav, sr=16000, n_steps=n_steps)
    return wav.astype(np.float32)

class CombinedVoiceDataset(Dataset):
    def __init__(self, indices, train=False):
        self.indices = indices
        self.train = train
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx_ = self.indices[i]
        import librosa
        wav, _ = librosa.load(wav_paths[idx_], sr=16000, mono=True)
        wav = wav[:MAX_LEN]
        if self.train:
            wav = augment_waveform(wav)
        feats = feat_extractor(wav, sampling_rate=16000, max_length=MAX_LEN,
                                padding="max_length", truncation=True, return_tensors="pt")
        return {"input_values": feats["input_values"][0], "labels": wav_labels[idx_]}

voice_train_ds = CombinedVoiceDataset(train_idx, train=True)
voice_test_ds  = CombinedVoiceDataset(test_idx, train=False)

voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=len(LABELS),
    id2label={i: n for i, n in enumerate(LABELS)}, label2id=label2id
).to(device)

def voice_collate(batch):
    return {"input_values": torch.stack([b["input_values"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch])}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_t.to(logits.device), label_smoothing=0.05)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def voice_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

best_acc_overall = 0.0

print("\n-- Phase 1: frozen encoder --")
voice_model.freeze_feature_encoder()
phase1_args = TrainingArguments(
    output_dir=DRIVE_CKPT + "_phase1",
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
    per_device_train_batch_size=16, per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=6, learning_rate=3e-5, warmup_ratio=0.1,
    logging_steps=50, report_to=[], fp16=torch.cuda.is_available()
)
phase1_trainer = WeightedTrainer(
    model=voice_model, args=phase1_args,
    train_dataset=voice_train_ds, eval_dataset=voice_test_ds,
    data_collator=voice_collate, compute_metrics=voice_compute_metrics
)
phase1_trainer.train()
phase1_metrics = phase1_trainer.evaluate()
print("Phase 1 metrics:", phase1_metrics)
best_acc_overall = max(best_acc_overall, phase1_metrics["eval_accuracy"])
voice_model.save_pretrained(FINAL_DIR); feat_extractor.save_pretrained(FINAL_DIR)

print("\n-- Phase 2: unfrozen encoder, full fine-tune --")
for p in voice_model.parameters():
    p.requires_grad = True
phase2_args = TrainingArguments(
    output_dir=DRIVE_CKPT + "_phase2",
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=8, learning_rate=5e-6, warmup_ratio=0.1,
    logging_steps=50, report_to=[], fp16=torch.cuda.is_available()
)
phase2_trainer = WeightedTrainer(
    model=voice_model, args=phase2_args,
    train_dataset=voice_train_ds, eval_dataset=voice_test_ds,
    data_collator=voice_collate, compute_metrics=voice_compute_metrics
)
phase2_trainer.train()
phase2_metrics = phase2_trainer.evaluate()
print("Phase 2 metrics:", phase2_metrics)

if phase2_metrics["eval_accuracy"] > best_acc_overall:
    best_acc_overall = phase2_metrics["eval_accuracy"]
    voice_model.save_pretrained(FINAL_DIR)
    feat_extractor.save_pretrained(FINAL_DIR)
    print("Phase 2 improved on phase 1 — saved as the final model.")
else:
    print("Phase 2 did not beat phase 1 — phase 1's model remains saved as the final model.")

print("\n-- Multi-window TTA evaluation --")
final_model = Wav2Vec2ForSequenceClassification.from_pretrained(FINAL_DIR).to(device).eval()

def tta_predict(wav_path):
    import librosa
    wav, _ = librosa.load(wav_path, sr=16000, mono=True)
    total_len = len(wav)
    win = MAX_LEN
    if total_len <= win:
        starts = [0]
    else:
        starts = [0, max(0, (total_len-win)//2), max(0, total_len-win)]
    probs_sum = None
    for s in starts:
        chunk = wav[s:s+win]
        feats = feat_extractor(chunk, sampling_rate=16000, max_length=win,
                                padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = F.softmax(final_model(**feats).logits, dim=-1)[0]
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / len(starts)).argmax().item()

correct = 0
for i in test_idx:
    pred = tta_predict(wav_paths[i])
    if pred == wav_labels[i]:
        correct += 1
tta_acc = correct / len(test_idx)
print(f"\n================ DONE ================")
print(f"Best single-window accuracy: {best_acc_overall:.4f}")
print(f"Multi-window TTA accuracy:   {tta_acc:.4f}")
print(f"Saved to: {FINAL_DIR}")
