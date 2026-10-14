# ============================================================
# PART 1 — TEXT: DistilBERT on dair-ai/emotion, saves DIRECT to Drive
# ============================================================
!pip install -q -U "transformers>=4.46,<5.0" "datasets>=3.0" "accelerate>=0.34" "evaluate>=0.4.3"

import os, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from google.colab import drive

drive.mount('/content/drive')
os.makedirs("/content/drive/MyDrive/signalwell-models", exist_ok=True)
DRIVE_CKPT = "/content/drive/MyDrive/signalwell-models/text_ckpt"
FINAL_DIR = "/content/drive/MyDrive/signalwell-models/text_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

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
    output_dir=DRIVE_CKPT,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    num_train_epochs=4, learning_rate=2e-5, weight_decay=0.01,
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

text_model.save_pretrained(FINAL_DIR)
tok.save_pretrained(FINAL_DIR)
print("Saved to Drive:", FINAL_DIR)
