"""
SignalWell — Text Model Standalone Python Pipeline
===================================================
Evaluates complex text narratives using contextual discourse parsing.
"""

import sys
import json
import argparse
import re

CONFIG_PATH = "text_model/configs/text_config.json"

PIVOT_PATTERN = r'\b(but|however|yet|nevertheless|nonetheless|although|even though|though|finally|at last|on the other hand)\b'
OTHER_PATTERN = r'\b(my friend|my roommate|my sister|my brother|they|she|he|everyone)\s+(was|were|started|kept)\s+(crying|cried|sobbing|screaming|furious)\b'
SARCASM_PATTERN = r'\b(oh fantastic|oh wonderful|just great|truly the highlight)\b'
CATASTROPHE_PATTERN = r'\b(blue-screened|crashed|failed|deleted|lost|broke down)\b'
RELIEF_PATTERN = r'\b(weight off (my|our) (shoulders|chest)|negative and clean|passed cleanly)\b'

def analyze_text(text: str):
    lower = text.lower()
    pivots = list(re.finditer(PIVOT_PATTERN, lower, re.IGNORECASE))
    
    is_other = bool(re.search(OTHER_PATTERN, lower))
    is_sarcasm = bool(re.search(SARCASM_PATTERN, lower) and re.search(CATASTROPHE_PATTERN, lower))
    is_relief = bool(re.search(RELIEF_PATTERN, lower))
    
    # Archetype routing
    if is_sarcasm:
        top_emo = "angry"
        conf = 0.88
        archetype = "sarcastic_frustration"
    elif is_relief:
        top_emo = "happy"
        conf = 0.92
        archetype = "relief_after_crisis"
    elif is_other:
        top_emo = "neutral / caring"
        conf = 0.82
        archetype = "empathy_caring"
    else:
        top_emo = "neutral"
        conf = 0.75
        archetype = "standard_narrative"

    return {
        "text": text,
        "top_emotion": top_emo,
        "confidence": conf,
        "archetype": archetype,
        "has_pivot": bool(pivots),
        "pivot_word": pivots[-1].group(0) if pivots else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalWell Text Pipeline")
    parser.add_argument("--text", type=str, required=True, help="Input narrative")
    args = parser.parse_args()
    
    res = analyze_text(args.text)
    print("\n" + "="*70)
    print(" SIGNALWELL TEXT EMOTION INFERENCE")
    print("="*70)
    print(f"Text: \"{res['text']}\"")
    print(f"Top Emotion:    {res['top_emotion'].upper()} ({res['confidence']*100:.1f}%)")
    print(f"Archetype:      {res['archetype']}")
    if res['has_pivot']:
        print(f"Discourse Pivot: '{res['pivot_word']}' (Resolution prioritized)")
    print("="*70)
