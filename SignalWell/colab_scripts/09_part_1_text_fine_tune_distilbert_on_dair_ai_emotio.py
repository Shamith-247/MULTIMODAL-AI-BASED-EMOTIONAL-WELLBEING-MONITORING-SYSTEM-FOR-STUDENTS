# ============================================================
# PART 1 — TEXT: fine-tune DistilBERT on dair-ai/emotion (6 classes)
# Includes GPU check + auto-save to Google Drive at the end
# ============================================================

!pip -q install "transformers>=4.42" "datasets>=2.19" accelerate evaluate --upgrade

import os, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# ---- GPU check ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cpu":
    print("\n*** WARNING: no GPU detected. This will be very slow (hours instead of minutes). ***")
    print("*** Go to Runtime > Change runtime type > T4 GPU, then Runtime > Disconnect and ***")
    print("*** delete runtime, reconnect, and rerun this cell before continuing.           ***\n")

os.makedirs("/content/outputs", exist_ok=True)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

print("=== [1/3] TEXT — fine-tuning DistilBERT on dair-ai/emotion ===")

text_ds = load_dataset("dair-ai/emotion", "split")
label_names_text = text_ds["train"].features["label"].names
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
print("\n================ DONE ================")
print("TEXT test metrics:", text_test_metrics)

text_model.save_pretrained("/content/outputs/text_model")
tok.save_pretrained("/content/outputs/text_model")
print("Saved locally to /content/outputs/text_model")

# ---- Auto-save to Google Drive ----
from google.colab import drive
drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
os.system("cp -r /content/outputs/text_model /content/drive/MyDrive/signalwell-models/")
print("Saved to Google Drive: /content/drive/MyDrive/signalwell-models/text_model")
