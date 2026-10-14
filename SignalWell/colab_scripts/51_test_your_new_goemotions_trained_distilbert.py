# ============================================================
# TEST — your new GoEmotions-trained DistilBERT
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_DIR = "/content/drive/MyDrive/signalwell-models/text_model_goemotions"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

print("Model loaded. Labels:", model.config.id2label)

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
    top5 = torch.topk(probs, 5)

    print(f"\nText: {text}")
    for score, idx in zip(top5.values, top5.indices):
        label = model.config.id2label[idx.item()]
        bar = "█" * int(score.item() * 30)
        print(f"    {label:<15} {score.item():.3f}  {bar}")
