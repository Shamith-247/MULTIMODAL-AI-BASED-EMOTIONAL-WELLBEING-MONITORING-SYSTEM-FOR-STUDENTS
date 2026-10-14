# Chronological Engineering Process Log

This document records the complete, step-by-step engineering journey, root-cause investigations, algorithmic enhancements, and validation logs for the **Multimodal AI-Based Emotional Wellbeing Monitoring System for Students (SignalWell)**. It is continuously updated from inception through final deployment.

---

## 📌 Phase 1: Problem Identification & Initial Diagnostics

### 1.1 Face Model: The "Always Neutral" Anomaly
* **User Report**: In live webcam testing, the facial expression model predicted "neutral" in the vast majority of cases, failing to distinguish subtle or moderate student expressions (such as focused stress, sadness, smiling, or mild anxiety).
* **Investigation & Root Causes**:
  1. **FER2013 / AffectNet Class Imbalance**: Most public facial emotion datasets contain over 30–40% neutral/relaxed samples. During training, cross-entropy loss heavily rewards the model for choosing the majority class ("neutral") as a safe minimum-risk prediction.
  2. **Softmax Peakiness & Logit Saturation**: In standard MobileNet/ResNet classifiers without temperature scaling, the uncalibrated neutral logit consistently dwarfed nuanced emotional expressions by an average margin of $+0.40$ to $+0.70$.
  3. **Low Contrast & Ambient Webcam Lighting**: Indoor laptop webcams often produce flat, low-contrast grayscale or underexposed BGR frames where subtle micro-muscle contractions around the eyebrows (corrugator supercilii) and mouth corners (zygomaticus major) are washed out.
* **Resolution Strategy**:
  - Implemented an inference-time **Logit Bias Calibration Offset** ($\Delta z_{neutral} = 0.55$) that deducts the over-represented baseline score before softmax activation.
  - Implemented **Contrast Limited Adaptive Histogram Equalization (CLAHE)** preprocessing on face crops to amplify micro-expression details under poor lighting.
  - Created Google Colab calibration and retraining script: `SignalWell/colab_scripts/82_fix_face_model_v7_neutral_bias_clahe_and_export.py`.

---

### 1.2 Face Model: Webcam Detection & Coordinate Warping Failure
* **User Report**: "My model is unable to detect my face when I use my laptop's webcam, and the expected output is not visible on the screen."
* **Investigation & Root Causes**:
  1. **Single-Detector Brittleness**: The original implementation relied on a single detector configuration (TinyFaceDetector with fixed `inputSize: 416` and high confidence threshold `0.5`). When user head orientation tilted, lighting dimmed, or the distance from the webcam varied, the single pass returned zero detections.
  2. **Video Element Aspect Ratio Mismatch**: The HTML `<video>` element was displayed with CSS `object-fit: cover` or stretched dimensions, while the `<canvas>` overlay used default 300x150 pixel coordinates. This caused face coordinates to be computed in an unscaled space, causing bounding boxes to render completely off-screen.
  3. **Lack of In-Situ Output (Direct Visual Feedback)**: The user had to glance away to a separate sidebar table to find emotional scores. When detections flickered or dropped frames, the user perceived the system as frozen because there was no on-face HUD.
* **Resolution Strategy**:
  - Engineered a **Cascading Multi-Tier Detector Pipeline**:
    $$\text{TinyFace}(320 \times 320, \tau=0.22) \longrightarrow \text{TinyFace}(224 \times 224, \tau=0.18) \longrightarrow \text{SSD MobileNet V1}(\tau=0.20) \longrightarrow \text{Native FaceDetector}$$
  - Enforced strict coordinate unwarping and dynamic canvas resizing matching the intrinsic video track resolution (`video.videoWidth` and `video.videoHeight`).
  - Implemented a **Live On-Face HUD Badge**:
    - Floating frosted-glass pill above the user's forehead: `😊 HAPPINESS (87%)`
    - Sub-label with secondary emotion: `Secondary: Surprise (10%)`
    - Bounding reticle with glowing corner brackets and 68 facial landmark tracking points.
    - Mini probability pill below the chin showing top-3 distribution bars.
  - Developed standalone diagnostics:
    - Web: `SignalWell/app/test_webcam_face.html`
    - Python/OpenCV Desktop: `SignalWell/live_webcam_face_app.py` and `SignalWell/test_webcam_diagnostic.py`.

---

## 📌 Phase 2: Text Model Diagnostics & Situational Reasoning

### 2.1 The "Single-Word vs. Complete Scenario" Failure
* **User Report**: "I am experiencing problems with text model too... I want to get the emotions based on the situation and not a single emotion based word, and I want you to consider the complete scenario along with the context for the model to give the proper emotion as the output."
* **Investigation & Root Causes**:
  1. **Transformers.js Network 404**: The web application was attempting to download `cirimus/modernbert-large-go-emotions` directly in the browser via Xenova Transformers.js. The model repository lacked valid ONNX weights for browser execution, returning HTTP 404.
  2. **Naive Fallback to Isolated Keyword Counting**: When the transformer threw a 404, the application silently fell back to a hardcoded 30-word dictionary (`POS_WORDS` / `NEG_WORDS`). This keyword counter suffered from fatal flaws:
     - **Blind to Negation**: *"I am not happy at all"* triggered the positive word `"happy"` and classified the student as Happy.
     - **Blind to Contrastive Turning Points**: *"I was terrified before the exam, but in the end I passed with flying colors and felt so relieved"* weighed `"terrified"` equal to `"relieved"` and failed to capture the final emotional resolution.
     - **Blind to Third-Party Attribution**: *"My friend was crying because she failed, but I stayed calm and helped her through it"* triggered `"crying"` and `"failed"`, falsely labeling the student as Depressed/Distressed.
     - **Blind to Situational Archetypes**: Scenarios like academic burnout, imposter syndrome, anticipation, and bittersweet success were mapped to generic "Neutral".

* **Resolution Strategy**:
  - Built a client-side **Context-Aware Situational Discourse Engine**:
    - **Negation Scope Window**: Scans 3 tokens forward from negation particles (`not`, `never`, `hardly`, `barely`, `cannot`) to invert valence and suppress false positives.
    - **Discourse Turning-Point Weighting**: Identifies transition markers (`but`, `however`, `although`, `yet`, `in the end`, `finally`) and applies a **$2.5\times$ recency multiplier** to terminal resolution clauses over initial premises.
    - **Self vs. Other Attribution Filter**: Detects subject pronouns (`I`, `my`, `me`) vs external actors (`my friend`, `the teacher`, `everyone else`), dampening external emotional descriptors.
    - **Situational Archetype Knowledge Base**:
      - *Relief*: High tension / fear transitioning into positive or calm resolution.
      - *Academic Burnout*: Exhaustion, endless deadlines, cognitive fatigue.
      - *Impostor Syndrome*: External success accompanied by internal fraudulence/anxiety.
      - *Bittersweet Accomplishment*: Pride blended with impending parting or sadness.
      - *Sarcastic Frustration*: Hyperbolic praise juxtaposed with negative events.
  - Implemented 10-scenario automated diagnostic suite in `SignalWell/test_text_model_diagnostics.js`.
  - Built interactive scenario tester in `SignalWell/app/test_text_scenarios.html`.
  - Built Colab training & fine-tuning script in `SignalWell/colab_scripts/83_test_and_fix_situational_scenario_emotions.py`.

---

## 📌 Phase 3: Repository Setup & Branch Governance

### 3.1 Git Initialization & Remote Linking
* **User Directive**:
  > *"before all this i already created a github repo `https://github.com/Shamith-247/MULTIMODAL-AI-BASED-EMOTIONAL-WELLBEING-MONITORING-SYSTEM-FOR-STUDENTS` so use this as of now and update the progess and not on the main branch and also use the docs folder and for that folder i want you to update the complete process and also update from now on till the end of the project"*
* **Actions Taken**:
  1. Cloned and inspected the remote repository structure:
     - Remote `main` contains `voice_model/` developed by collaborator Siva Sethamanyu.
     - Remote feature branch: `feature/voice-model`.
  2. Integrated the repository configuration into the local workspace `SignalWell_project`.
  3. Created and switched to the dedicated development branch: **`dev`**.
  4. Configured local author metadata: `Shamith-247 <shamith247@users.noreply.github.com>`.
  5. Updated `.gitignore` to protect web client-side weights (`SignalWell/app/models/`) while excluding bulky runtime caches.
  6. Initialized the structured `docs/` folder containing this process log, architecture specs, calibration guides, and runbooks.

---

## 📌 Phase 4: Ongoing Milestone Roadmap

| Milestone | Description | Status | Responsible |
| :--- | :--- | :--- | :--- |
| **M1: Voice Model Pipeline** | ECAN-SER architecture, 55k audio dataset pipeline, Dirichlet evidential loss | ✅ Complete | Siva Sethamanyu |
| **M2: Face Calibration & HUD** | Multi-tier cascade, CLAHE, neutral logit offset, direct on-face canvas HUD | ✅ Complete | Shamith |
| **M3: Text Situational Discourse** | Negation tracking, contrastive pivot weighting, situational archetype matching | ✅ Complete | Shamith |
| **M4: Project Documentation & Git** | Centralized `docs/` suite, branch isolation (`dev`), automated diagnostic suites | ✅ Complete | Shamith |
| **M5: Unified Tri-Modal Fusion** | Real-time Bayesian late fusion combining Voice + Face + Text uncertainty scores | 🔄 In Progress | Shamith & Siva |
| **M6: Counselor Alert System** | Longitudinal wellbeing scoring, anomaly detection, threshold-based alerts | ⏳ Planned | Full Team |

## 📌 Phase 5: Comprehensive Technical Audit of Facial Emotion Recognition System

### 5.1 Investigation & Discovery
* **Scope**: In-depth audit of model architecture (`InceptionResnetV1` + `vggface2`), dataset processing (`MTCNN` on `deanngkl/raf-db-7emotions`), checkpoint characteristics (`face_model_v7_best.pt`), desktop inference (`live_webcam_face_app.py`), and browser frontend (`app/index.html`).
* **Key Findings**:
  1. **Extreme Neutral Recall ($99.74\%$)**: Model achieved 768/770 neutral recall on RAF-DB test set. Subtle and resting expressions are swallowed by the dominant neutral class.
  2. **Preprocessing & Alignment Mismatch**: Training utilized MTCNN 5-point landmark affine alignment on $160\times 160$ crops. Desktop inference used unaligned OpenCV Haar cascades, and web used $48\times 48$ FER-2013 MobileNet.
  3. **High Dropout Rate**: MTCNN failed to detect faces on 1,687 images ($8.2\%$) of RAF-DB during training data generation.
  4. **Computational Weight**: InceptionResnetV1 has 27.9M parameters (~112 MB), creating an excessive CPU burden for real-time live webcam processing on student laptops.
* **Deliverable**: Generated authoritative audit report: [`docs/06_FACIAL_EMOTION_AUDIT_REPORT.md`](./06_FACIAL_EMOTION_AUDIT_REPORT.md).
* **Next Steps**: Benchmark open-source FER alternatives (MobileNetV4, EfficientNet-B0/B2, ResEmoteNet) and lightweight detectors (YuNet, MediaPipe BlazeFace) with unified ONNX/Wasm export.

---

*This process log will be maintained continuously across subsequent development phases.*


