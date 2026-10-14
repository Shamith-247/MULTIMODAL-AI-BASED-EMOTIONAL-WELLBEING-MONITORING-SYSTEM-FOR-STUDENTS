# ============================================================
# COMPARE — your model vs. GoEmotions (Reddit-trained, 28 labels)
# ============================================================
!pip install -q transformers

from transformers import pipeline
import torch

go_emotions = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

test_sentences = [
    "I can't believe how great today turned out, everything just worked.",
    "I've been feeling really overwhelmed and exhausted lately.",
    "Honestly I'm just going through the motions today, nothing special.",
    "I miss my family so much, I wish they were here right now.",
]

for text in test_sentences:
    results = go_emotions(text)[0]
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    print(f"\nText: {text}")
    for r in results_sorted:
        bar = "█" * int(r["score"] * 30)
        print(f"    {r['label']:<15} {r['score']:.3f}  {bar}")
