# ============================================================
#  VOICE MODEL — Wav2Vec2 on RAVDESS+CREMA-D+TESS+SAVEE
#  Checkpoints to Drive DURING training so a disconnect doesn't lose everything
# ============================================================

!pip -q install "transformers>=4.42" "datasets>=2.19" accelerate evaluate librosa soundfile kaggle --upgrade

import os, json, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
                           TrainingArguments, Trainer)
import evaluate
from google.colab import drive

# ---- Mount Drive FIRST, before anything else ----
drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
DRIVE_CKPT_DIR = "/content/drive/MyDrive/signalwell-models/voice_ckpt_v2"

# ---- >>> REPLACE THESE TWO VALUES WITH YOUR OWN KAGGLE CREDENTIALS <<< ----
KAGGLE_USERNAME = "your_kaggle_username_here"
KAGGLE_KEY = "your_kaggle_api_key_here"
# ----------------------------------------------------------------------------

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
print("Kaggle credentials set")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cpu":
    print("*** WARNING: no GPU — check Runtime > Change runtime type > T4 GPU, then reconnect ***")

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
    p = path.replace("\\", "/")
    lower = p.lower()
    fname = os.path.basename(p)
    if "ravdess" in lower:
        parts = fname.replace(".wav", "").split("-")
        if len(parts) >= 3 and parts[2] in RAVDESS_CODE:
            return RAVDESS_CODE[parts[2]]
        return None
    if "crema" in lower:
        parts = fname.replace(".wav", "").split("_")
        if len(parts) >= 3 and parts[2] in CREMA_CODE:
            return CREMA_CODE[parts[2]]
        return None
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
                wav_paths.append(full)
                wav_labels.append(label2id[lab])

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

class CombinedVoiceDataset(Dataset):
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

voice_train_ds = CombinedVoiceDataset(train_idx)
voice_test_ds  = CombinedVoiceDataset(test_idx)

voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=len(LABELS),
    id2label={i: n for i, n in enumerate(LABELS)}, label2id=label2id
).to(device)
voice_model.freeze_feature_encoder()

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

# ---- KEY CHANGE: output_dir points STRAIGHT INTO DRIVE, and it checkpoints every epoch ----
voice_args = TrainingArguments(
    output_dir=DRIVE_CKPT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",      # <- saves a real checkpoint to Drive after every epoch
    save_total_limit=2,         # <- keeps only the 2 most recent, so Drive doesn't fill up
    per_device_train_batch_size=16, per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=6, learning_rate=3e-5, warmup_ratio=0.1,
    logging_steps=50, report_to=[], fp16=torch.cuda.is_available()
)

voice_trainer = WeightedTrainer(
    model=voice_model, args=voice_args,
    train_dataset=voice_train_ds, eval_dataset=voice_test_ds,
    data_collator=voice_collate, compute_metrics=voice_compute_metrics
)
voice_trainer.train()
voice_test_metrics = voice_trainer.evaluate()
print("\n================ DONE ================")
print("VOICE v2 test metrics:", voice_test_metrics)

# ---- Final clean save, directly to Drive ----
final_dir = "/content/drive/MyDrive/signalwell-models/voice_model_v2"
voice_model.save_pretrained(final_dir)
feat_extractor.save_pretrained(final_dir)
print("Saved to Google Drive:", final_dir)
