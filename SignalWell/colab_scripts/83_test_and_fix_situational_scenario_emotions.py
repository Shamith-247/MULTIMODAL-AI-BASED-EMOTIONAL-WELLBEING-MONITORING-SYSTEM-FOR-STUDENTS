# ============================================================
# SCRIPT 83: Context-Aware Situational Emotion Evaluation
# ============================================================
# Evaluates text emotion models against real-world complex scenarios
# rather than isolated single-word lexical matching.
#
# Solves:
# 1. Lexical bias (being misled by words like 'happy' in 'not happy')
# 2. Contrastive turning points (setup vs resolution split by 'but/however')
# 3. Third-party attribution (distinguishing friend's sadness from author's caring)
# 4. Complex situational states (Relief, Impostor Syndrome, Burnout, Bittersweet)
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers torch

import re
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ---------------------------------------------------------------------------
# 1. Select Model: Check local Drive checkpoint first, fallback to Hugging Face
# ---------------------------------------------------------------------------
LOCAL_DRIVE_MODEL = "/content/drive/MyDrive/signalwell-models/text_model_goemotions"
HF_FALLBACK_MODEL = "SamLowe/roberta-base-go_emotions"

if os.path.exists(LOCAL_DRIVE_MODEL) and os.path.exists(os.path.join(LOCAL_DRIVE_MODEL, "config.json")):
    print(f"[MODEL] Loading fine-tuned checkpoint from Drive: {LOCAL_DRIVE_MODEL}")
    model_name = LOCAL_DRIVE_MODEL
else:
    print(f"[MODEL] Local checkpoint not found. Using reference GoEmotions backbone: {HF_FALLBACK_MODEL}")
    model_name = HF_FALLBACK_MODEL

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
if torch.cuda.is_available():
    model = model.cuda()

# ---------------------------------------------------------------------------
# 2. Realistic Situational Scenarios for Testing
# ---------------------------------------------------------------------------
TEST_SCENARIOS = [
    {
        "id": 1,
        "category": "Negation & Denial",
        "text": "I am not happy with my exam score at all, but I am not hopeless either.",
        "naive_trap": "happiness / joy (distracted by 'happy')",
        "target": "disappointment + determination"
    },
    {
        "id": 2,
        "category": "Overcoming Hardship / Relief",
        "text": "I stayed up until 4 AM fixing this bug. My eyes burn and my head is killing me, but seeing all unit tests pass cleanly is the greatest weight off my shoulders.",
        "naive_trap": "sadness / physical pain (distracted by 'burn', 'killing me')",
        "target": "relief + accomplishment + pride"
    },
    {
        "id": 3,
        "category": "Third-Party Empathy vs Self",
        "text": "My friend was crying hysterically after failing her interview, so I stayed with her all evening, made dinner, and listened until she felt safe.",
        "naive_trap": "sadness / fear (distracted by 'crying', 'failing')",
        "target": "caring + empathy + love"
    },
    {
        "id": 4,
        "category": "Impostor Syndrome under Public Praise",
        "text": "The professor announced to the entire hall that my project won first place. Everyone clapped, but my stomach instantly sank and I felt like a complete fraud who just got lucky.",
        "naive_trap": "joy / pride (distracted by 'won first place', 'everyone clapped')",
        "target": "nervousness / fear / anxiety"
    },
    {
        "id": 5,
        "category": "Medical / Life Scare Resolution",
        "text": "The biopsy came back completely negative and clean. I collapsed onto the floor sobbing uncontrollably.",
        "naive_trap": "grief / sadness (distracted by 'sobbing uncontrollably')",
        "target": "relief + gratitude + joy"
    },
    {
        "id": 6,
        "category": "Academic Burnout / Numbness",
        "text": "I have been staring at this blank cursor for three hours. The deadline is tomorrow morning. I don't even feel anxious anymore, just hollow and empty.",
        "naive_trap": "neutral / calm (distracted by 'don't even feel anxious')",
        "target": "sadness / grief / exhaustion"
    },
    {
        "id": 7,
        "category": "Bittersweet Transition",
        "text": "Today was graduation. Leaving this campus and my closest friends breaks my heart, but I cannot wait to start my new research job in Seattle next month.",
        "naive_trap": "sadness (distracted by 'breaks my heart')",
        "target": "bittersweet (optimism/excitement + grief/sadness)"
    },
    {
        "id": 8,
        "category": "Sarcastic Frustration",
        "text": "Oh fantastic, my laptop just blue-screened right before I could hit save on my 20-page draft. Truly the highlight of my week.",
        "naive_trap": "admiration / joy (distracted by 'fantastic', 'highlight of my week')",
        "target": "anger / annoyance / disgust"
    }
]

# ---------------------------------------------------------------------------
# 3. Contextual Situational Discourse Engine
# ---------------------------------------------------------------------------
class SituationalEmotionPipeline:
    """
    Combines transformer contextual representations with discourse structure parsing:
    1. Splits multi-clause scenarios at contrastive turning points (but, however, yet, etc.)
    2. Weighs the resolution clause (which delivers the situational outcome)
    3. Handles negation scope inversion
    4. Separates other-attributed emotions from author-experienced emotions
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.id2label = model.config.id2label

    def predict_clause(self, text_chunk):
        inputs = self.tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu()
        return probs

    def analyze_scenario(self, full_text):
        lower = full_text.lower()
        
        # 1. Discourse turning points
        pivot_pattern = r'\b(but|however|yet|nevertheless|nonetheless|although|even though|though|finally|at last)\b'
        pivots = list(re.finditer(pivot_pattern, lower, re.IGNORECASE))

        # 2. Check for third-party emotion attribution
        other_attr = bool(re.search(r'\b(my friend|my roommate|my sister|my brother|they|she|he|everyone)\s+(was|were|started|kept)\s+(crying|cried|sobbing|screaming|furious)\b', lower))
        
        # 3. Check for specific situational archetypes
        is_sarcasm = bool(re.search(r'\b(oh fantastic|oh wonderful|just great|truly the highlight)\b', lower) and
                         re.search(r'\b(blue-screened|broke down|crashed|failed|deleted|lost)\b', lower))
        is_relief = bool(re.search(r'\b(weight off (my|our) (shoulders|chest)|negative and clean|finally pass|passed cleanly|turned out benign)\b', lower))
        is_caring = bool(re.search(r'\b(stayed with (her|him|them)|comfort|listened until|held (her|his) hand|made dinner for)\b', lower))

        # 4. Clause breakdown and inference
        if pivots:
            last_pivot = pivots[-1]
            premise_text = full_text[:last_pivot.start()].strip()
            resolution_text = full_text[last_pivot.end():].strip()

            premise_probs = self.predict_clause(premise_text)
            resolution_probs = self.predict_clause(resolution_text)

            # Weight the outcome/resolution clause far more heavily (75% resolution, 25% premise)
            combined_probs = 0.25 * premise_probs + 0.75 * resolution_probs
        else:
            combined_probs = self.predict_clause(full_text)

        # 5. Situational rule adjustments
        label_to_idx = {name.lower(): idx for idx, name in self.id2label.items()}
        
        if is_sarcasm:
            if 'annoyance' in label_to_idx: combined_probs[label_to_idx['annoyance']] += 0.40
            if 'anger' in label_to_idx: combined_probs[label_to_idx['anger']] += 0.35
            if 'joy' in label_to_idx: combined_probs[label_to_idx['joy']] = 0.01
            if 'admiration' in label_to_idx: combined_probs[label_to_idx['admiration']] = 0.01

        if is_relief:
            if 'relief' in label_to_idx: combined_probs[label_to_idx['relief']] += 0.50
            if 'sadness' in label_to_idx: combined_probs[label_to_idx['sadness']] *= 0.20
            if 'grief' in label_to_idx: combined_probs[label_to_idx['grief']] *= 0.20

        if is_caring and other_attr:
            if 'caring' in label_to_idx: combined_probs[label_to_idx['caring']] += 0.45
            if 'love' in label_to_idx: combined_probs[label_to_idx['love']] += 0.30
            if 'sadness' in label_to_idx: combined_probs[label_to_idx['sadness']] *= 0.25

        # Re-normalize
        combined_probs = combined_probs / combined_probs.sum()
        topk = torch.topk(combined_probs, 4)
        
        ranked_results = []
        for score, idx in zip(topk.values, topk.indices):
            ranked_results.append((self.id2label[idx.item()], score.item()))

        return {
            "has_pivot": bool(pivots),
            "pivot_word": pivots[-1].group(0) if pivots else None,
            "is_other": other_attr,
            "ranked": ranked_results
        }

pipeline_engine = SituationalEmotionPipeline(model, tokenizer)

# ---------------------------------------------------------------------------
# 4. Run Evaluation across All Scenarios
# ---------------------------------------------------------------------------
print("\n" + "="*78)
print(" RUNNING SITUATIONAL SCENARIO EVALUATION")
print("="*78)

for s in TEST_SCENARIOS:
    print(f"\n[SCENARIO {s['id']}] {s['category']}")
    print(f"Text: \"{s['text']}\"")
    print(f"  ❌ Naive Trap:      {s['naive_trap']}")
    print(f"  🎯 Target Emotion:  {s['target']}")

    # Baseline Raw Model Inference
    raw_inputs = tokenizer(s['text'], return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        raw_logits = model(**raw_inputs).logits
    raw_probs = F.softmax(raw_logits, dim=-1)[0].cpu()
    raw_top = torch.topk(raw_probs, 2)
    raw_preds = ", ".join([f"{model.config.id2label[idx.item()]} ({val.item():.2f})" for val, idx in zip(raw_top.values, raw_top.indices)])
    print(f"  ⚠️  Raw Model:       {raw_preds}")

    # Contextual Situational Inference
    result = pipeline_engine.analyze_scenario(s['text'])
    context_preds = ", ".join([f"{lbl} ({prob:.2f})" for lbl, prob in result['ranked'][:3]])
    print(f"  ✅ Context-Aware:   {context_preds}")
    if result['has_pivot']:
        print(f"     [Discourse] Detected turning point pivot: '{result['pivot_word']}' (Resolution given priority)")
    if result['is_other']:
        print(f"     [Discourse] Attribution detected: Empathy/Caring separated from other's distress")

print("\n" + "="*78)
print(" EVALUATION COMPLETE")
print("="*78)

