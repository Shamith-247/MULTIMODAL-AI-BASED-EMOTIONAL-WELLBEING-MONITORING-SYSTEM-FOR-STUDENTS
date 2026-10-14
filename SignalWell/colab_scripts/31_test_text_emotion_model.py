# ============================================================
# TEST — Text emotion model
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tok = AutoTokenizer.from_pretrained("/content/drive/MyDrive/signalwell-models/text_model")
model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/signalwell-models/text_model").eval()

# ---- Edit this to test different sentences ----
test_sentences = [
    "I can't believe how great today turned out, everything just worked.",
    "I've been feeling really overwhelmed and exhausted lately.",
    "Honestly I'm just going through the motions today, nothing special.",
]

for text in test_sentences:
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    top_idx = probs.argmax().item()
    print(f"\nText: {text}")
    print(f"  Predicted: {model.config.id2label[top_idx]} ({probs[top_idx]*100:.1f}% confidence)")
    for i, p in enumerate(probs):
        print(f"    {model.config.id2label[i]}: {p.item():.3f}")
