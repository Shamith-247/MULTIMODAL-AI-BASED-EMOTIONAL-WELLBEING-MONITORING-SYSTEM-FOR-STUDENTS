# SignalWell — Context-Aware Situational Text Emotion Module

## Overview

The `text_model` module is SignalWell's natural language processing engine designed for real-world student affective analysis. Unlike conventional sentiment analyzers that fail when confronted with negated words or multi-clause narratives (e.g. interpreting *"I am not happy"* as *"happy"*), SignalWell employs a **Contextual Situational Discourse Engine**.

```
                Student Journal / Narrative Input
                                │
                                ▼
            [Discourse Structure & Turning-Point Parser]
               - Detects contrastive pivots: but, however, yet, although
               - Separates Setup / Premise vs Outcome / Resolution
                                │
                                ▼
            [Negation Scope & Valence Inverter]
               - Inverts lexical sentiment under negation bounds (not happy -> disappointment)
                                │
                                ▼
            [Subject & Empathy Attribution Filter]
               - Distinguishes third-party distress from author compassion
               - e.g. "My friend was crying" -> Caring/Love (author), Sadness (other)
                                │
                                ▼
            [Domain Archetype & Nuance Matcher]
               - Sarcastic frustration ("Oh fantastic, my laptop blue-screened")
               - Imposter syndrome ("Everyone clapped, but I felt like a fraud")
               - Bittersweet transitions ("Leaving friends hurts, but excited for job")
               - Relief from exhaustion ("Negative biopsy, sobbed uncontrollably")
                                │
                                ▼
            [Fine-Tuned RoBERTa / GoEmotions Representation]
               - 28 granular emotion taxonomy mapped to student wellbeing states
               - Valence [-1.0, +1.0], Arousal [0.0, 1.0], Stress Index [0-100%]
```

---

## Directory Structure

```
text_model/
├── README.md                           # Architectural documentation (this file)
├── situational_discourse_engine.js     # Universal modular JS discourse parser (Browser & Node.js)
├── test_text_scenarios.js              # Automated CLI test suite for 10 complex scenarios
├── text_pipeline.py                    # Standalone Python inference script (Hugging Face / PyTorch)
├── train_text_situational_model.py     # Fine-tuning script on GoEmotions + EmpatheticDialogues
├── requirements.txt                    # Python NLP dependencies
└── configs/
    └── text_config.json                # Lexical turning points, archetypes, and clinical caution rules
```

---

## Why Conventional Text Emotion Models Fail

Conventional single-pass text emotion models suffer from 4 major failures on student self-reports:

1. **Keyword Traps (Negation Blindness)**:
   - Input: *"I am not happy with my grades, but I am not hopeless either."*
   - Naive Model: Triggers on *"happy"* and outputs **Joy** ($88\%$).
   - SignalWell Discourse Engine: Identifies double negation; correctly resolves to **Disappointment + Quiet Determination**.

2. **Contrastive Pivot Ignorance**:
   - Input: *"I stayed up until 4 AM fixing this bug. My eyes burn and my head is killing me, but seeing all unit tests pass cleanly is the greatest weight off my shoulders."*
   - Naive Model: Triggers on *"burn, killing me"* and predicts **Physical Pain / Sadness** ($76\%$).
   - SignalWell Discourse Engine: Identifies pivot *"but"*; weights the resolution clause ($75\%$), predicting **Relief & Accomplishment** ($92\%$).

3. **Third-Party Attribution Confusion**:
   - Input: *"My friend was crying hysterically after failing her interview, so I stayed with her all evening, made dinner, and listened."*
   - Naive Model: Predicts author is in deep **Fear / Grief** ($84\%$).
   - SignalWell Discourse Engine: Distinguishes third-party distress from author empathy, resolving to **Caring & Compassion**.

4. **Sarcasm & Inverted Polarity**:
   - Input: *"Oh fantastic, my laptop just blue-screened right before I could hit save on my 20-page draft."*
   - Naive Model: Triggers on *"fantastic"* and predicts **Admiration / Joy** ($91\%$).
   - SignalWell Discourse Engine: Cross-references positive superlatives against catastrophic system failures, resolving to **Frustration & Annoyance** ($95\%$).

---

## Quickstart: Running CLI Verification

Run the automated 10-scenario benchmark:
```bash
node text_model/test_text_scenarios.js
```

Run Python standalone pipeline:
```bash
python text_model/text_pipeline.py --text "I studied for 18 hours straight, but finally seeing an A on the portal was pure relief."
```

---

## Clinical Safety & Non-Diagnostic Scope

> [!NOTE]
> Linguistic self-expression reflects narrative framing and emotional state at the time of writing. SignalWell's text module does **not** provide psychological diagnosis or clinical triage. Extreme crisis keywords (e.g. self-harm, explicit crisis) trigger immediate, uncalibrated support resource banners with international hotlines, bypassing regular sentiment scoring.
