# Technical Audit Report: SignalWell Facial Emotion Recognition System

**Author**: Senior Computer Vision & MLOps Engineering Team  
**Date**: October 2026  
**Status**: Authoritative System Audit & Architectural Review  
**Branch**: `dev`

---

## Executive Summary

This audit delivers an exhaustive technical assessment of the facial emotion recognition component within the **SignalWell** multimodal wellbeing monitoring system. We audited the full engineering stack: from Google Colab training scripts (versions 00 through 83) and historical training artifacts (`attachment.txt`, `attachment-3ed85a15.txt`) to local desktop inference (`live_webcam_face_app.py`) and browser interfaces (`app/index.html`).

While the training pipeline achieved **88.05% test accuracy** and **78.16% Macro-F1** on the held-out split of RAF-DB, the real-world webcam inference has proven unreliable in live testing. Our audit identifies the fundamental root causes:
1. **Severe Neutral Attraction**: The model achieved **99.74% recall on the neutral class**, effectively monopolizing low-to-moderate intensity facial expressions.
2. **Training vs. Inference Alignment Mismatch**: The model was trained on tightly cropped, 5-point landmark-aligned faces using MTCNN, but live desktop inference relied on unaligned OpenCV Haar cascades, and the browser interface ran an unrelated 48x48 MobileNet (FER-2013).
3. **Detector Dropout**: MTCNN failed to detect faces and dropped **1,687 images (~8.2%)** from the RAF-DB dataset during training, and is too computationally heavy (180–350ms on CPU) for smooth live webcam inference.
4. **Architectural Misalignment**: `InceptionResnetV1` (27.9M parameters, 110 MB) was designed for identity verification—an objective whose representation space actively seeks invariance to facial expressions.

Below is the structured 10-point audit report followed by concrete architectural recommendations.

---

## Section 1: Existing Architecture

```mermaid
flowchart TD
    subgraph ColabTraining ["Colab Training Pipeline (v7)"]
        A[RAF-DB Dataset: 20,471 images] --> B[MTCNN Face Detector & 5-pt Alignment]
        B -->|Dropped 1,687 failed crops| C[160x160 RGB Face Crops]
        C --> D[Data Augmentation: Flip, Rot 8°, ColorJitter]
        D --> E[InceptionResnetV1 Backbone: 27.9M params]
        E --> F[nn.Dropout 0.3 + nn.Linear 512->7]
        F --> G[CrossEntropyLoss with Class Weights]
    end

    subgraph BrowserInference ["Browser Dashboard (app/index.html)"]
        H[Live HTML5 Webcam Video] --> I[Cascading Detector: TinyFace 320/224 + SSD MobileNet]
        I --> J[Face-api.js ExpressionNet: 48x48 MobileNet FER2013]
        J --> K[In-Situ On-Face HUD Canvas Overlay]
    end

    subgraph DesktopInference ["Desktop Script (live_webcam_face_app.py)"]
        L[OpenCV DirectShow Capture] --> M[OpenCV Haar Cascade Face Detector]
        M --> N[CLAHE Lighting Correction]
        N --> O[Tries to load face_model_v7_best.pt / ONNX]
        O --> P[Fallback to Geometric Heuristics if absent]
    end
```

### Detailed Component Specifications:
* **Backbone**: `InceptionResnetV1` from `facenet-pytorch`, initialized with weights pretrained on `vggface2` (3.3M images, 9,131 identities).
* **Classification Head**: `nn.Dropout(0.3)` followed by `nn.Linear(512, 7)`.
* **Parameter Count**: $27,913,918$ parameters (~112 MB for FP32 weights).
* **Target Classes**: 7 basic emotions: `anger`, `disgust`, `fear`, `happiness`, `neutral`, `sadness`, `surprise`.
* **Input Resolution**: $160 \times 160 \times 3$ (RGB).
* **Normalization**: $\text{pixel} \in [0, 1] \rightarrow \frac{\text{pixel} - 0.5}{0.5} \in [-1, 1]$.
* **Training Protocol**: 3-Phase transfer learning:
  - Phase 1 (6 epochs): Frozen backbone, train classification head only ($\text{lr} = 10^{-3}$, OneCycleLR).
  - Phase 2 (8 epochs): Unfreeze block8, repeat_3, last_linear, last_bn ($\text{lr} = 3 \times 10^{-4}$, OneCycleLR).
  - Phase 3 (10 epochs): Full fine-tuning ($\text{lr} = 3 \times 10^{-5}$, CosineAnnealingLR).

---

## Section 2: What Currently Works

1. **Browser Video Ingestion & Hardware Interfacing**:
   The HTML5 `navigator.mediaDevices.getUserMedia` pipeline successfully opens local webcams at native resolution (720p/1080p) across modern browsers.
2. **In-Situ On-Face HUD Canvas**:
   The visual presentation tier (`app/index.html` and `app/test_webcam_face.html`) accurately tracks facial coordinates, dynamically draws neon corner brackets, renders a 68-point landmark mesh, and displays a floating frosted-glass badge (`😊 HAPPINESS (87%)`) with under-chin probability distribution bars.
3. **Context-Aware Situational Text Engine**:
   The client-side discourse analyzer in `SignalWell/` operates cleanly with zero external API dependencies, passing 10/10 student scenarios including negations, contrastive pivots, and third-party attribution filters.
4. **Speech Emotion Recognition Pipeline**:
   The ECAN-SER model in `voice_model/` with evidential Dirichlet loss is self-contained and functions as intended.
5. **Colab Training Convergence**:
   The training implementation in `colab_scripts/` converges reliably without gradient explosions, achieving high validation accuracy ($89.39\%$) on benchmark image splits.

---

## Section 3: What Fails

1. **Severe Neutral Attraction in Live Webcam Tests**:
   Users report getting "neutral at most of the cases". In the test evaluation log (`attachment.txt`), the recall on the neutral class reached an extreme **0.9974** (768 out of 770 test samples classified as neutral). The model's decision boundaries for neutral have expanded to encompass resting faces, mild smiles, and concentrated expressions.
2. **Detector Dropout During Dataset Preprocessing**:
   MTCNN failed to detect faces and dropped **1,209 train**, **252 validation**, and **226 test** images from RAF-DB. This discarded $8.2\%$ of the training data—specifically samples with intense emotional grimaces, head tilts, or partial occlusions.
3. **Disconnection Between PyTorch Checkpoint and Web Dashboard**:
   The browser app (`index.html`) relies on `face-api.js` (an old client-side MobileNet trained on FER-2013). The user's trained `face_model_v7_best.pt` model is never loaded or executed in the web dashboard.
4. **Desktop Python Environment Barriers**:
   The local Windows machine lacks a native Python installation in PATH (Python points to the Windows Store alias), preventing direct local execution of PyTorch or OpenCV scripts without manual environment setup.
5. **Haar Cascade vs. MTCNN Bounding Box Incompatibility**:
   `live_webcam_face_app.py` falls back to OpenCV Haar cascades for detection. Haar cascades produce unaligned, jittery square crops that mismatch MTCNN's 5-point affine-aligned crops, degrading InceptionResnetV1's feature representations.

---

## Section 4: Why Current Webcam Inference Is Unreliable

1. **Domain Shift (Acted Benchmarks vs. Real Laptop Webcams)**:
   RAF-DB photos feature front-facing, well-illuminated portraits. Laptop webcams are situated at low angles, frequently suffer from harsh backlighting (windows/lamps), and exhibit sensor noise.
2. **CPU Inference Latency of MTCNN**:
   MTCNN consists of three cascaded deep networks (P-Net, R-Net, O-Net) that construct an image pyramid. On standard laptop CPUs, MTCNN takes 180–350 ms per frame, limiting video throughput to 3–5 FPS and causing stutter.
3. **Absence of Temporal Smoothing**:
   Raw frame-by-frame softmax inference suffers from high variance due to micro-movements, lighting shifts, and eye blinks. Without exponential smoothing, predictions flicker erratically.

---

## Section 5: Dataset Suitability & Task Alignment

* **Training Dataset**: `deanngkl/raf-db-7emotions` (20,471 images).
* **Label Distribution**:
  - Happiness: $5,957$ ($29.1\%$)
  - Neutral: $5,132$ ($25.1\%$)
  - Anger: $4,071$ ($19.9\%$)
  - Sadness: $2,460$ ($12.0\%$)
  - Surprise: $1,619$ ($7.9\%$)
  - Disgust: $877$ ($4.3\%$)
  - Fear: $355$ ($1.7\%$)
  - **Imbalance Ratio**: $16.78\times$ between Happiness and Fear.
* **Task Mismatch**:
  RAF-DB captures acute, exaggerated facial expressions. However, in student wellbeing monitoring, students rarely display high-arousal expressions like "Fear" or "Disgust". Student distress manifests as prolonged cognitive exhaustion, flat affect, furrowed brows, or masked tension. Using discrete 7-class FER as a direct proxy for emotional wellbeing is scientifically and clinically invalid without longitudinal temporal tracking.

---

## Section 6: Checkpoint & Architecture Compatibility

* The checkpoint format in `face_model_v7_best.pt`:
  - Contains full `state_dict` matching `InceptionResnetV1(pretrained='vggface2')` + `Dropout(0.3)` + `Linear(512, 7)`.
  - Embedded metadata: `labels`, `val_acc: 0.8939`, `test_acc: 0.8805`, `macro_f1: 0.7816`.
* **Status in Local Repo**: The `.pt` file is stored in Google Drive (`/content/drive/MyDrive/signalwell-models/face_model_v7_best.pt`) and is not present in the local repository directory.
* **Compatibility**: The PyTorch definition in `live_webcam_face_app.py` correctly matches the checkpoint weights, but the web browser dashboard cannot load PyTorch `.pt` files without an ONNX/TF.js bridge.

---

## Section 7: Training vs. Inference Preprocessing Comparison

| Step | Training Pipeline (Colab v7) | Desktop Script (`live_webcam_face_app.py`) | Web App (`index.html`) | Alignment Status |
| :--- | :--- | :--- | :--- | :---: |
| **Color Space** | RGB (PIL Image) | BGR converted to RGB | RGB (HTML Video Canvas) | ✅ Aligned |
| **Face Detector** | MTCNN (5-point affine alignment) | OpenCV Haar Cascade (No alignment) | TinyFaceDetector / SSD MobileNet | ❌ **MISMATCH** |
| **Crop Margin** | `margin=20` pixels | Variable bounding box | Tight bounding box | ❌ **MISMATCH** |
| **Input Size** | $160 \times 160$ | $160 \times 160$ | $48 \times 48$ / $112 \times 112$ | ❌ **MISMATCH** |
| **Normalization** | $(x - 0.5) / 0.5 \in [-1, 1]$ | $(x - 0.5) / 0.5 \in [-1, 1]$ | Standard ImageNet / $[0, 1]$ | ⚠️ Divergent |
| **Lighting Equalization** | None (Raw pixels) | CLAHE on L-channel (LAB) | Dynamic canvas contrast | ⚠️ Divergent |

---

## Section 8: Primary Performance Bottlenecks

1. **InceptionResnetV1 Parameter Size**: $27.9\text{M}$ parameters requires $>5.7\text{ GFLOPs}$ per inference pass, consuming significant CPU resources on student hardware.
2. **MTCNN Multi-Stage Pyramid**: Ineffective for real-time video on CPU.
3. **Single-Frame Jitter**: Absence of temporal smoothing causes flickering classifications.

---

## Section 9: Components Worth Retaining

1. **The In-Situ On-Face HUD Canvas**: Excellent user experience and visual clarity.
2. **The Context-Aware Text Situational Engine**: High accuracy ($10/10$) on complex student narrative scenarios.
3. **The Voice SER Pipeline (`voice_model/`)**: ECAN-SER with evidential uncertainty is sound.
4. **The 3-Phase Training Protocol**: Head-only $\rightarrow$ Partial unfreeze $\rightarrow$ Full fine-tuning is an established transfer learning practice.
5. **Class-Weighted Cross-Entropy Loss & Macro-F1 Metric Tracking**.

---

## Section 10: Components to Optimize or Replace

1. **Replace MTCNN with Modern Ultra-Lightweight Face Detectors**:
   - For Python: **YuNet (OpenCV DNN)** or **MediaPipe BlazeFace** (under $1\text{ MB}$, $>100\text{ FPS}$ on CPU, built-in 5-point landmark alignment).
   - For Web: **MediaPipe Face Detection** / **TinyFaceDetector**.
2. **Replace/Complement InceptionResnetV1 with a Dedicated FER Architecture**:
   - Candidate A: **MobileNetV4 / MobileNetV3-Small FER** ($3\text{M}$ parameters, $12\text{ MB}$, $>60\text{ FPS}$ on CPU, easily exportable to ONNX and WebAssembly).
   - Candidate B: **EfficientNet-B0 / B2 FER** ($5\text{M}$ parameters, state-of-the-art accuracy on AffectNet/RAF-DB).
   - Candidate C: **ResEmoteNet** (Residual Squeeze-and-Excitation CNN designed specifically for facial expression recognition).
3. **Implement Full ONNX / WebAssembly Export**:
   Provide a single unified model format that runs across desktop Python and browser dashboards without disparity.
4. **Integrate Exponential Moving Average (EMA) Temporal Smoothing**:
   $$\mathbf{s}_t = \alpha \mathbf{s}_{t-1} + (1 - \alpha) \mathbf{p}_t, \quad \alpha = 0.70$$
   to eliminate frame-to-frame visual flickering.
5. **Calibrated Temperature & Neutral Logit Scaling**:
   Dampen the over-represented neutral class dynamically based on lighting and landmark motion variance.

