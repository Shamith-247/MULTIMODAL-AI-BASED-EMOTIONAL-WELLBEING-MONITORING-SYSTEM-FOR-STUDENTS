# ============================================================
# TEST — Text emotion model (with diagnostics)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

TEXT_MODEL_DIR = "/content/drive/MyDrive/signalwell-models/text_model"

print("Folder exists:", os.path.exists(TEXT_MODEL_DIR))
if os.path.exists(TEXT_MODEL_DIR):
    print("Folder contents:", os.listdir(TEXT_MODEL_DIR))
else:
    raise FileNotFoundError(f"{TEXT_MODEL_DIR} doesn't exist — check the folder name in Drive")

required_files = ["config.json"]
missing = [f for f in required_files if not os.path.exists(os.path.join(TEXT_MODEL_DIR, f))]
if missing:
    raise FileNotFoundError(f"Missing required file(s) in {TEXT_MODEL_DIR}: {missing} — "
                             f"the model wasn't saved correctly here, it needs retraining/resaving")

tok = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR, local_files_only=True).eval()

print("Model loaded successfully. Labels:", model.config.id2label)

test_sentences = [
    "I can't believe how great today turned out, everything just worked.",
    "I've been feeling really overwhelmed and exhausted lately.",
    "Honestly I'm just going through the motions today, nothing special.",
    "I miss my family so much, I wish they were here right now.",
]

for text in test_sentences:
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    top_idx = probs.argmax().item()

    print(f"\nText: {text}")
    print(f"  Predicted: {model.config.id2label[top_idx]}  ({probs[top_idx]*100:.1f}% confidence)")
    print("  Full breakdown:")
    for i, p in enumerate(probs):
        bar = "█" * int(p.item() * 30)
        print(f"    {model.config.id2label[i]:<10} {p.item():.3f}  {bar}")
