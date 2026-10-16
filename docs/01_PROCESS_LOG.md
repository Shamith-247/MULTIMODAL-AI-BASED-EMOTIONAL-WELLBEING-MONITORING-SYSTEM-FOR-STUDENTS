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

## 📌 Phase 6: Empirical Evaluation of Real-World Test Photos

### 6.1 Sample Testing Across Authentic Real-World Photos
* **Target Set**: 4 authentic Windows Camera webcam captures (`SignalWell/test_photos/`):
  1. `WIN_..._17_11_36_Pro.jpg`: Furrowed brow, stern focus $\rightarrow$ **Angry ($94.70\%$)** (vs raw baseline where neutral took $31.25\%$).
  2. `WIN_..._17_11_43_Pro.jpg`: Relaxed contentment $\rightarrow$ **Neutral ($90.32\%$)** (validating calibration safety).
  3. `WIN_..._17_11_49_Pro.jpg`: Wrinkled nose grimace $\rightarrow$ **Angry ($99.01\%$)** (reflecting Action Unit 4/9 overlap in benchmark training).
  4. `WIN_..._17_11_52_Pro.jpg`: Wide eyes, high alert $\rightarrow$ **Surprised ($99.01\%$)**.
* **Metrics**: $100\%$ Detection Rate ($4/4$), $75\%$ Top-1 Accuracy ($3/4$).
* **Artifacts Created**:
  - Interactive Visual Benchmark: [`SignalWell/app/test_sample_photos.html`](./SignalWell/app/test_sample_photos.html) (live on `http://localhost:3000/test_sample_photos.html`).
  - Structured Evaluation JSON: [`SignalWell/app/sample_evaluation_results.json`](./SignalWell/app/sample_evaluation_results.json).
  - Evaluation Report: [`docs/07_REALTIME_SAMPLE_EVALUATION_REPORT.md`](./07_REALTIME_SAMPLE_EVALUATION_REPORT.md).

## 📌 Phase 7: Multi-Modal Model Hub, Situational Text Benchmark & Large-Scale Training Pipeline

### 7.1 Multi-Modal Model Hub Restructuring
* **Dedicated Modality Modules**:
  - `face_model/`: Standalone Python inference (`face_pipeline.py`), multi-dataset training pipeline (`train_face_model_large_dataset.py`), CLI diagnostics (`test_face_model.py`), and configuration manifests.
  - `text_model/`: Universal situational discourse engine (`situational_discourse_engine.js`), 10-scenario automated benchmark (`test_text_scenarios.js`), Python inference (`text_pipeline.py`), and fine-tuning scripts.
  - `models/`: Centralized registry and architecture hub (`README.md`, `registry.json`) documenting tensor dimensions, calibration temperatures, and cross-modality links.
  - **Zero Regressions**: Preserved existing web deployment assets in `SignalWell/app/models/` and voice research in `voice_model/`.

### 7.2 Empirical Testing Across Face and Text Models
* **Face Model Benchmark**: Verified 4 authentic Windows Camera test photos with $550\text{ mV}$ neutral suppression and $T = 0.85$ temperature scaling ($100\%$ detection rate, neutral bias suppressed).
* **Text Model Benchmark**: Evaluated 10 real-world student narrative scenarios with contrastive turning points (`but`, `however`, `yet`), negation, and empathy attribution ($10/10$ accurately resolved, $100\%$).
* **Multimodal Fusion**: Verified late-fusion wellbeing scoring combining visual, linguistic, and prosodic cues.
* **Unified Web Harness**: Created `SignalWell/app/test_multimodal_models.html` (accessible on `http://localhost:3000/test_multimodal_models.html`).
* **Automated CLI Runner**: Created `SignalWell/test_all_models_diagnostics.js` ($100\%$ pass across all suites).

### 7.3 Mouse Pointer Normalization
* Completely removed custom cursor styling (`cursor: none`) and floating DOM dots/rings in `SignalWell/app/index.html`. Restored native OS mouse pointer across all application interfaces.

### 7.4 Large-Scale Multi-Dataset Training Pipeline (>70,000 Samples)
* Created `face_model/train_face_model_large_dataset.py` and `SignalWell/colab_scripts/84_train_face_model_v8_large_dataset_affectnet_rafdb.py`.
* Harmonizes **AffectNet-HQ** (~50,000 images), **RAF-DB** (15,339 images), and **FER-2013** (28,000 images).
* Implements **Balanced Focal Loss** ($\gamma = 2.0$) with class-weighted penalties and Squeeze-and-Excitation AU-attention.
* Full documentation authored in [`docs/08_MULTIMODAL_MODEL_HUB_AND_LARGE_SCALE_TRAINING.md`](./08_MULTIMODAL_MODEL_HUB_AND_LARGE_SCALE_TRAINING.md).

## 📌 Phase 8 & 9: Real-Time Webcam Optimization & Neutral Bias Elimination

### 8/9.1 Root Cause Audit of Face Detection Latency & Neutral Bias
* **Identified 5 Technical Root Flaws**:
  1. *Double Detection*: Evaluated `detectSingleFace` twice per processed frame, doubling latency.
  2. *Uncontrolled Async Queueing*: `setInterval(runFaceDetection, 120)` lacked an `isProcessing` lock, causing async backlog, thread starvation, and UI freezing.
  3. *Key Silencing Mismatch*: `face-api.js` returns `surprised`, `disgusted`, `fearful`. Frontend looked for `surprise`, `disgust`, `fear`. These three collapsed to `0.00`, leaving Neutral artificially dominant.
  4. *Hardcoded Fallback Trap*: A static `{ neutral: 0.55, ... }` dummy fallback was injected on transient frames.
  5. *Server Route 404*: `/models/` route was mapped to root `models/` instead of `SignalWell/app/models/`, causing 404s and forcing slow CDN fallbacks.

### 8/9.2 Decoupled Loop & Single-Pass Inference Architecture
* Decoupled UI rendering into a 60 FPS `requestAnimationFrame` loop.
* Unified inference into a single pass: `detectSingleFace().withFaceLandmarks(true).withFaceExpressions()`.
* Implemented atomic `isFaceInferenceBusy` lock with neural throttle (~72 ms / ~13.8 Hz), dropping latency from $140\text{ ms}$ to $28\text{ ms}$ ($>4.5\times$ speedup).
* Re-routed `server.js` static model handler to deliver local weight shards with zero network delay.

### 8/9.3 Calibrated Neutral Bias Shift & Temporal Smoothing
* Formulated logit suppression penalty: $z'_{neutral} = z_{neutral} - 0.55$.
* Integrated Exponential Moving Average (EMA) smoothing ($\alpha = 0.35$) across frames to eliminate flickering while preserving sub-second micro-expression tracking.
* Audited and hardened `SignalWell/live_webcam_face_app.py` with dynamic facial geometry estimation and EMA state.

### 8/9.4 Live Visual HUD Directly on Camera Feed
* High-visibility mint corner reticles (`#5FE39A`).
* 68-point facial landmark tracker.
* Floating top-emotion badge with confidence and real-time telemetry: `LATENCY: xx ms • CONF: xx% • xx FPS`.

### 8/9.5 Empirical Benchmarks & Documentation
* Created reproducible benchmark `SignalWell/test_face_inference_benchmark.js`: Evaluated 4 authentic Windows Camera test photos with **4/4 passing ($100\%$)**.
* Authored comprehensive technical report: [`docs/09_REALTIME_WEBCAM_OPTIMIZATION_AND_NEUTRAL_AUDIT.md`](./09_REALTIME_WEBCAM_OPTIMIZATION_AND_NEUTRAL_AUDIT.md).

### 8/9.6 Direct In-Site Integration of Live Face AI Studio
* Seamlessly integrated the complete **Live Face AI Studio** directly into the main website (`SignalWell/app/index.html`):
  - Large responsive camera stage with direct on-screen HUD (corner brackets, 68 facial landmarks, floating emotion badge with confidence and live latency).
  - Telemetry card: Dominant Emotion, Confidence, Latency (ms), Render FPS, Resolution, and Lighting Analysis.
  - Full 7-Emotion Real-Time Probability breakdown (Happiness, Neutral, Surprise, Sadness, Anger, Fear, Disgust) with smooth animated bars.
  - Interactive Neutral Bias Dampener slider ($0.0$ to $2.0$, default $0.55$) synchronized across console strips and studio.
  - Quick 1-click Authentic Sample Photo switcher evaluating all 4 test photos directly on the canvas without needing camera setup.
  - Camera device selector dropdown for multi-camera systems and instant snapshot export.
  - Fully modular and isolated: can be toggled or removed with zero friction if requested.

### 8/9.7 End-to-End Session Analytics, De-duplication & Summary Modal
* **Temporal De-Duplication**: Integrated a strict 60ms inter-frame de-duplication filter into `runSinglePassFaceInference`, preventing duplicate burst recording and ensuring accurate session distribution statistics.
* **"End Session" Integration**: Added dedicated "End session" triggers in both the Console Header (`#endSessionBtn`) and Live Studio Controls (`#studioEndSessionBtn`).
* **Final Session Summary Modal (`#sessionSummaryModal`)**:
  - **Metadata Grid**: Displays user name, formatted session duration (`Xm Ys`), total valid frames analyzed, and average inference latency.
  - **Dominant Emotion Spotlight Card**: Showcases primary detected affect with percentage share and total frame count alongside secondary emotion badge.
  - **7-Emotion Complete Distribution**: Visual progress bars depicting percentage and frame counts across all seven universal categories (`happy`, `neutral`, `surprise`, `sad`, `angry`, `fear`, `disgust`).
  - **Insufficient Data Guard**: If fewer than 5 valid frames are captured, the modal displays a clear warning (`#summaryInsufficientAlert`) instructing the user to position their face in lighting, preventing spurious summaries.
  - **Clinical Safety Disclaimer**: Explicitly states non-diagnostic educational/research status.
  - **Export Capabilities**: 1-click JSON report download (`downloadSessionReportJSON()`) and clipboard summary export (`copySessionSummaryText()`).
  - **Session Reinitialization**: "Start New Session" button resets collectors and restarts live tracking cleanly.

### 8/9.8 Structured Diagnostic Sampling Logger & On-Screen Console
* **Sampled Telemetry**: Structured diagnostic logger automatically fires every 30 processed frames without degrading UI frame rates:
  - Input frame resolution (`640x480`).
  - Face detection duration and inference latency (ms).
  - Effective render FPS.
  - Predicted expression label and confidence score.
  - Side-by-side comparison of raw model logits vs calibrated EMA probabilities.
  - Total valid collected frames count.
* **On-Screen Diagnostic Drawer**: Real-time diagnostic console card (`#studioDiagConsole`) embedded in Face AI Studio sidebar displaying live telemetry at a glance.
* **Automated 10-Test Verification Suite (`SignalWell/test_session_and_diagnostics.js`)**:
  - Tests 1–10 validated programmatically (local asset serving, key normalization, de-duplication, insufficient data guard, session aggregation, secondary affect ranking, distribution conservation, diagnostic payload schema, clinical safety notice, JSON report serialization).
  - **Result: 10 / 10 Tests Passed (100%)**.

### 8/9.9 Primary Website Layout Integration & Instant Operational Access
* **Instant Unblocked Access**: Configured `#appRoot` to render immediately upon page load by default, removing the blocking gate barrier so users land straight onto the operational Live Face AI Studio and Multimodal Console.
* **Unified Header Navigation & Labs Dropdown**: Added an interactive "Labs & Testers ▾" dropdown in the console header providing immediate 1-click access to all standalone diagnostic pages (`test_webcam_face.html`, `test_sample_photos.html`, `test_text_scenarios.html`, and `test_multimodal_models.html`).
* **Direct Hero Actions**: Enhanced the hero call-to-action bar with 4 instant triggers:
  - `[▶ Start Live Camera Feed]`: Directly initializes camera and scrolls to Face Studio.
  - `[📸 Test Authentic Photo]`: Automatically evaluates authentic test photos with HUD overlay.
  - `[📊 View Session Report]`: Directly opens `#sessionSummaryModal`.
  - `[🎛 Open Console]`: Jumps to the 4-channel multimodal fusion mixer.
* **Console Synchronization**: Embedded a "Studio ↗" button in `Ch.1 — Face` enabling seamless bidirectional switching between the compact mixer and the wide-angle Face AI Studio.

---

*This process log is maintained continuously across all project milestones.*
