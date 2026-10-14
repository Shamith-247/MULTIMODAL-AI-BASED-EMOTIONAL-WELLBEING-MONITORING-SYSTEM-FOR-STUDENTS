# ============================================================
# WIDER TEST — GoEmotions model across diverse journal-style sentences
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_DIR = "/content/drive/MyDrive/signalwell-models/text_model_goemotions"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

# A broader, more realistic spread of journal-style entries —
# mixed emotions, mundane moments, anxiety, gratitude, anger, ambiguity, fragments.
test_sentences = [
    # Clear positive
    "Had coffee with an old friend today, it felt so good to catch up.",
    "Finally finished the project I've been putting off for weeks.",
    # Clear negative
    "I keep replaying that conversation in my head and it's driving me crazy.",
    "Nothing seems to be going right this week.",
    # Mixed / ambivalent
    "Proud of myself for finishing the race but my legs are killing me.",
    "Excited about the new job but honestly a little scared too.",
    # Mundane / low-stakes
    "Did laundry, made dinner, watched some TV. Pretty normal day.",
    "Woke up late again, need to fix my sleep schedule.",
    # Anxiety-leaning
    "Can't stop worrying about the presentation tomorrow.",
    "My heart's been racing all day for no real reason.",
    # Anger-leaning
    "So annoyed that they cancelled last minute again.",
    "I'm furious about how that meeting went.",
    # Gratitude-leaning
    "Really grateful for my sister checking in on me today.",
    "Small things like a good cup of tea make such a difference.",
    # Grief/loss-adjacent
    "It's been a year since he passed and it still hits me sometimes.",
    # Short fragments (common in real journaling)
    "Tired. Just tired.",
    "Okay day I guess.",
    "Actually kind of a good one today :)",
]

for text in test_sentences:
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    top3 = torch.topk(probs, 3)

    print(f"\n{text}")
    for score, idx in zip(top3.values, top3.indices):
        label = model.config.id2label[idx.item()]
        bar = "█" * int(score.item() * 30)
        print(f"    {label:<15} {score.item():.3f}  {bar}")
