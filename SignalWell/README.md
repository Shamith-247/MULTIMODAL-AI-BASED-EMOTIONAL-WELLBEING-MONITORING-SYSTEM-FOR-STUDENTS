# SignalWell - project bundle

## Open first
- app/index.html - latest app. Camera needs http://localhost (e.g. `python -m http.server 8000` in app/, then open localhost:8000); getUserMedia is blocked on file://.
- demos/posture-demo.html, demos/geometric-face-demo.html - standalone tests (not wired into the app). Same serve-on-localhost rule for camera.

## Status
| Channel | Model | Result | In app |
|---|---|---|---|
| Text | ModernBERT-large GoEmotions (cirimus) | chosen after comparisons | yes |
| Face | v7 (InceptionResnetV1, RAF-DB) + faceExpressionNet | 88.05% test; neutral bias resolved via logit dampening (-1.3), contrast equalization, & resting calibration | yes (dual-engine: v7 ONNX + instant client fallback) |
| Voice | voice_model_v3 | ~85% | no (heuristic placeholder only) |
| Behavior | typing/session heuristic | unvalidated | yes |

## colab_scripts/
Every large code cell from the conversation in order (numbered). Many are successive iterations of the same thing; later numbers are newer. No credentials included - Kaggle/HF tokens are placeholders you fill in.
your_original_pastes/ holds code/outputs you pasted. test_photos/ are your webcam shots.
