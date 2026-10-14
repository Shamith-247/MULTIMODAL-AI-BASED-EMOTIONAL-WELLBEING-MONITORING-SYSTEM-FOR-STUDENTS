# Deployment, Testing & Verification Runbook

This runbook provides step-by-step instructions for launching, testing, and verifying every component of the **Multimodal AI Emotional Wellbeing Monitoring System for Students (SignalWell)** across both web and native desktop environments.

---

## 📋 System Prerequisites

* **Operating System**: Windows 10/11, macOS, or Linux.
* **Node.js**: v18.0.0 or higher.
* **Python**: v3.9+ (optional, for native OpenCV desktop testing and Colab pipelines).
* **Webcam**: Standard integrated 720p/1080p laptop camera or USB webcam.
* **Web Browser**: Google Chrome or Microsoft Edge (recommended for WebGL and hardware-accelerated MediaStream APIs).

---

## 🚀 Quick Start: Launching the Unified Multimodal Dashboard

### Step 1: Clone & Switch to Development Branch
Always ensure you are working on the active development branch:
```bash
git clone https://github.com/Shamith-247/MULTIMODAL-AI-BASED-EMOTIONAL-WELLBEING-MONITORING-SYSTEM-FOR-STUDENTS.git
cd MULTIMODAL-AI-BASED-EMOTIONAL-WELLBEING-MONITORING-SYSTEM-FOR-STUDENTS
git checkout dev
```

### Step 2: Start the Local Development Server
The project includes a lightweight zero-dependency Node.js HTTP server configured with correct MIME types for WebAssembly, neural model shards, and static assets:
```bash
node SignalWell/server.js
```
*Output:*
```
[SignalWell] Local server started at http://localhost:3000
[SignalWell] Serving:
  - Main Multimodal Dashboard:  http://localhost:3000/index.html
  - Standalone Webcam Face HUD: http://localhost:3000/test_webcam_face.html
  - Standalone Text Scenarios:  http://localhost:3000/test_text_scenarios.html
```

---

## 🧪 Testing Face Detection & On-Screen HUD

### Option A: Interactive Browser Tester (`test_webcam_face.html`)
1. Open Google Chrome and navigate to:
   ```
   http://localhost:3000/test_webcam_face.html
   ```
2. Click **"Allow"** when prompted for webcam access.
3. Observe the live HUD overlay:
   - **Glowing Reticle**: Neon corners frame your face.
   - **Landmark Mesh**: 68 subtle tracking dots track your facial contours.
   - **Floating Forehead Badge**: Displays your live emotion (e.g. `😊 HAPPINESS (84%)`).
   - **Under-Chin Distribution Pill**: Proportional bars for top-3 candidate emotions.
4. Test the **Neutral Logit Offset Slider**:
   - Slide between `0.00` and `0.80` to observe how neutral dampening allows subtle smiles and expressions to emerge without getting stuck in "Neutral".

### Option B: Native Desktop Python/OpenCV Application
For environments where browser hardware acceleration is unavailable, run the direct OpenCV desktop application:
```bash
python SignalWell/live_webcam_face_app.py
```
* **Features**:
  - Automatically queries camera indexes `0`, `1`, `2` with DirectShow backend.
  - Applies real-time CLAHE contrast enhancement.
  - Renders the floating on-screen HUD badge and confidence pills directly on OpenCV GUI frames.
  - Press `q` or `Esc` to cleanly terminate the stream.

---

## 📝 Testing Context-Aware Text Scenarios

### Option A: Automated CLI Diagnostic Suite
Run the 10-scenario regression test in your terminal:
```bash
node SignalWell/test_text_model_diagnostics.js
```
*Expected Output:*
```
======================================================================
 SIGNALWELL TEXT EMOTION & SITUATIONAL DISCOURSE DIAGNOSTIC SUITE
======================================================================
[TEST 1] "I am not happy with my grades at all."
  -> Primary: Sadness/Disappointment (84%) | Negation detected: YES | PASS

[TEST 2] "I was terrified before the viva, but in the end I aced it and feel so relieved!"
  -> Primary: Relief/Joy (88%) | Turning-point detected: YES | PASS

[TEST 3] "My friend was crying because she failed, but I stayed calm and supported her."
  -> Primary: Calm/Supportive (82%) | Third-party attribution: ACTIVE | PASS
...
======================================================================
RESULTS: 10 / 10 Tests Passed (100% Accuracy)
======================================================================
```

### Option B: Interactive Browser Scenario Sandbox
1. Navigate to:
   ```
   http://localhost:3000/test_text_scenarios.html
   ```
2. Click any of the 10 preset scenario buttons (e.g., *Academic Burnout*, *Impostor Syndrome*, *Relief After Anxiety*).
3. Type custom reflective student sentences to verify turning-point weights, negation handling, and primary emotion outputs.

---

## 🎙️ Testing Speech Emotion Recognition (Voice Pipeline)

The voice pipeline is located in `voice_model/`:
```bash
python voice_model/SIGNALWELL_VOICE_PIPELINE.py --audio voice_model/sample_1.wav
```
* Outputs predicted emotion probabilities, evidential Dirichlet confidence, and epistemic uncertainty $u_v$.

---

## ☁️ Google Colab Cloud GPU Pipelines

For retraining or fine-tuning neural weights on cloud GPUs:
1. **Face Model Neutral Bias Fix & CLAHE Export**:
   - Open Google Colab and upload:
     `SignalWell/colab_scripts/82_fix_face_model_v7_neutral_bias_clahe_and_export.py`
   - Select Runtime $\rightarrow$ Change runtime type $\rightarrow$ **T4 GPU**.
   - Runs focal loss rebalancing, applies CLAHE, and exports calibrated ONNX/TF.js shards.
2. **Text Situational Scenario Fine-Tuning**:
   - Upload `SignalWell/colab_scripts/83_test_and_fix_situational_scenario_emotions.py` to Colab.
   - Trains discourse-aware classifier on student narrative benchmarks.

---

## 🛡️ Git Governance & Branch Policy

To protect project stability and respect team collaboration:
* **STRICT RULE**: **Never push directly to `main`**.
* All experimental and multimodal feature commits must be pushed to `dev` or dedicated feature branches (e.g. `feature/*`).
* Pull requests will be created and reviewed before merging into `main`.

