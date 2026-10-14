# ============================================================
# TEST — Text emotion model
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

TEXT_MODEL_DIR = "/content/drive/MyDrive/signalwell-models/text_model"

tok = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR).eval()

print("Model loaded. Labels:", model.config.id2label)

# ---- Edit these sentences to test whatever you want ----
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
