from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

TEXT_MODEL_DIR = "/content/drive/MyDrive/signalwell-models/text_ckpt/checkpoint-2000"

# Tokenizer from the original base model — vocab is unchanged by fine-tuning
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Actual trained weights from your checkpoint
model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR, local_files_only=True).eval()

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
    top_idx = probs.argmax().item()
    print(f"\nText: {text}")
    print(f"  Predicted: {model.config.id2label[top_idx]}  ({probs[top_idx]*100:.1f}% confidence)")
    for i, p in enumerate(probs):
        bar = "█" * int(p.item() * 30)
        print(f"    {model.config.id2label[i]:<10} {p.item():.3f}  {bar}")

# If this looks right, save a clean complete copy immediately
tok.save_pretrained("/content/drive/MyDrive/signalwell-models/text_model")
model.save_pretrained("/content/drive/MyDrive/signalwell-models/text_model")
print("\nClean copy saved to signalwell-models/text_model")
