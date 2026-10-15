# SignalWell — Centralized Model Registry & Multi-Modal Architecture Hub

## Overview

The `models/` directory serves as the centralized registry and architecture catalog for all affective computing modules in SignalWell:

```
                            SignalWell Central Model Registry
                                           │
         ┌─────────────────────────────────┼────────────────────────────────┐
         ▼                                 ▼                                ▼
  [Face Model Hub]                  [Text Model Hub]                [Voice Model Hub]
  Location: face_model/             Location: text_model/           Location: voice_model/
  Architecture: MobileNetV4-FER     Architecture: RoBERTa-Discourse Architecture: ECAN-SER
  Classes: 7 Ekman                  Taxonomy: 28 GoEmotions         Classes: 7 Voice Emotions
  Input: 224x224x3 Face Crop        Input: Free-text Journal        Input: 16 kHz Audio Stream
```

In addition to this central registry, each modality maintains its own dedicated, clean directory (`face_model/`, `text_model/`, `voice_model/`), while existing web-deployment model weights in `SignalWell/app/models/` remain fully functional and intact for zero-friction browser execution.

---

## Model Inventory & Registry Specifications

| Modality | Architecture / Backbone | Checkpoint / Weights Format | Input Dimension | Primary Classes | Latency (CPU) |
|---|---|---|---|---|---|
| **Face** | MobileNetV4-FER + Squeeze-and-Excitation | `.pt`, `.onnx`, `model.json` (TF.js) | $224 \times 224 \times 3$ RGB | 7: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise | $<12$ ms |
| **Text** | RoBERTa-GoEmotions + Situational Discourse Engine | HuggingFace Hub / PyTorch checkpoint | Token sequence ($\le 512$) | 28 Granular + 7 Ekman Mappings | $<25$ ms |
| **Voice** | ECAN-SER (WavLM Large + BC-CA Conformer) | PyTorch `.pt` + ONNX | 16 kHz mono waveform (6s) | 7 Voice Emotions + Dirichlet Uncertainty | $<65$ ms |
| **Fusion** | Multi-Modal Bayesian Late Fusion | Client-side JS / WebAssembly | Vectors $[P_F, P_T, P_V]$ | Valence, Arousal, Stress, Fatigue | $<1$ ms |

---

## Machine-Readable Registry

All runtime configuration parameters, calibration temperatures, class index mappings, and source URIs are defined in [`registry.json`](registry.json).

### Example Loading in Python:
```python
import json

with open("models/registry.json", "r") as f:
    registry = json.load(f)

face_cfg = registry["models"]["face"]
print(f"Face model version: {face_cfg['version']}, Calibrated temperature: {face_cfg['calibration']['temperature']}")
```

---

## Directory Cross-References

- **Face Model Source & Scripts**: [`face_model/`](../face_model/)
- **Text Model Source & Scripts**: [`text_model/`](../text_model/)
- **Voice Model Source & Scripts**: [`voice_model/`](../voice_model/)
- **Browser Sharded Weights**: [`SignalWell/app/models/`](../SignalWell/app/models/)
- **Google Colab Training Notebooks**: [`SignalWell/colab_scripts/`](../SignalWell/colab_scripts/)
