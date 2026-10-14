# ============================================================
# PART 3 — VOICE: fine-tune Wav2Vec2 on RAVDESS (8 emotion classes)
# ============================================================
import os, zipfile, io, requests, random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
                           TrainingArguments, Trainer)
import evaluate

print("\n=== [3/3] VOICE — fine-tuning Wav2Vec2 on RAVDESS ===")

accuracy_metric = evaluate.load("accuracy")

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
print("Class distribution:", {n: wav_labels.count(i) for i, n in enumerate(label_names_voice)})

idx = list(range(len(wav_paths))); random.seed(42); random.shuffle(idx)
split = int(0.85 * len(idx))
train_idx, test_idx = idx[:split], idx[split:]

feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base", do_normalize=True)
MAX_LEN = 16000 * 4  # 4 seconds at 16kHz

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
voice_model.freeze_feature_encoder()  # keep CNN front-end frozen — faster, less overfitting on 1.4k clips

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
print("Saved to /content/outputs/voice_model")
