# ============================================================
# TEXT v2 — DistilBERT on GoEmotions (single-label subset, 28 classes)
# Saves DIRECT to Drive
# ============================================================
!pip install -q -U "transformers>=4.46,<5.0" "datasets>=3.0" "accelerate>=0.34" "evaluate>=0.4.3"

import os, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from google.colab import drive

drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
DRIVE_CKPT = "/content/drive/MyDrive/signalwell-models/text_goemotions_ckpt"
FINAL_DIR = "/content/drive/MyDrive/signalwell-models/text_model_goemotions"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# ---- Load GoEmotions, keep only single-label examples for a clean, comparable metric ----
raw_ds = load_dataset("go_emotions", "simplified")
label_names = raw_ds["train"].features["labels"].feature.names
print(f"GoEmotions has {len(label_names)} labels:", label_names)

def is_single_label(example):
    return len(example["labels"]) == 1

def extract_label(example):
    example["label"] = example["labels"][0]
    return example

filtered = {}
for split in ["train", "validation", "test"]:
    ds_split = raw_ds[split].filter(is_single_label)
    ds_split = ds_split.map(extract_label)
    filtered[split] = ds_split
    print(f"{split}: {len(ds_split)} single-label examples (out of {len(raw_ds[split])} total)")

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tok_fn(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=64)

tokenized = {}
for split in ["train", "validation", "test"]:
    t = filtered[split].map(tok_fn, batched=True)
    t = t.remove_columns(["text", "labels", "id"]).rename_column("label", "labels")
    t.set_format("torch")
    tokenized[split] = t

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(label_names),
    id2label={i: n for i, n in enumerate(label_names)},
    label2id={n: i for i, n in enumerate(label_names)}
).to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

args = TrainingArguments(
    output_dir=DRIVE_CKPT,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    num_train_epochs=4, learning_rate=2e-5, weight_decay=0.01,
    logging_steps=200, report_to=[], fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)
trainer.train()
test_metrics = trainer.evaluate(tokenized["test"])
print("\n================ DONE ================")
print("GoEmotions test metrics:", test_metrics)

model.save_pretrained(FINAL_DIR)
tok.save_pretrained(FINAL_DIR)
print("Saved to Drive:", FINAL_DIR)
