from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np, torch, evaluate

tok = AutoTokenizer.from_pretrained("/content/drive/MyDrive/signalwell-models/text_model")
model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/signalwell-models/text_model")
model.eval()

text_ds = load_dataset("dair-ai/emotion", "split")
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

preds, labels = [], []
for batch in text_ds["test"]:
    inputs = tok(batch["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds.append(logits.argmax(-1).item())
    labels.append(batch["label"])

print("accuracy:", accuracy_metric.compute(predictions=preds, references=labels))
print("f1_macro:", f1_metric.compute(predictions=preds, references=labels, average="macro"))
