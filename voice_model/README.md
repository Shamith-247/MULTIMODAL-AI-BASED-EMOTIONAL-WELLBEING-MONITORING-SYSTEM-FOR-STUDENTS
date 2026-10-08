# ======================================================================
# SignalWell Voice — Speech Emotion Recognition (SER) Module
# ======================================================================

A state-of-the-art dual-branch Speech Emotion Recognition (SER) pipeline combining **Self-Supervised Speech Representations (WavLM Large)** with **physiologically-grounded acoustic prosodic features (37-dimensional vector)** routed through a **Temporal Transformer Encoder** with dynamic **Attention Pooling** and **Uncertainty Quantification**.

---

## 🚀 Key Highlights & Architecture

* **Dual-Branch Input:**
  * **WavLM Large (SSL):** Sliced into 6 chronological 1-second chunks (`[Batch, 6, 1024]`).
  * **Acoustic Prosody (Physical):** 37-dimensional vector (F0 Pitch via YIN, RMS Energy, Zero Crossing Rate, Spectral Centroid/Bandwidth/Rolloff, Pause/Voicing ratio, 13 MFCC Means & Stds).
* **Fusion & Temporal Modeling:**
  * Joint projection to a shared 256-dimensional latent space.
  * 3-Layer Pre-LN Temporal Transformer (`nhead=8`, `dim_feedforward=1024`).
* **Learned Attention Pooling:** Additive attention aggregation over temporal chunks.
* **Dual Output Heads:**
  * **Emotion Classifier:** 7 classes (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
  * **Uncertainty Head:** Calibrated confidence scoring (0.0 to 1.0) flagging acoustic ambiguity.

---

## 📊 Benchmark Results

Evaluated on a **strict speaker-independent split** across 4 combined datasets (**RAVDESS, CREMA-D, TESS, and SAVEE**):

* **Test Accuracy:** `68.20%`
* **Test Macro-F1:** `68.32%`
* **Test UAR (Unweighted Average Recall):** `67.49%`
* **Validation Macro-F1:** `66.06%`

---

## 📁 Repository Structure

```
├── SIGNALWELL_VOICE_PIPELINE.py    # Complete standalone Python training & inference script
├── SIGNALWELL_VOICE_COLAB.ipynb    # All-in-one Google Colab single-cell notebook
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules (prevents large binary leaks)
├── sample_1.wav                    # Sample evaluation audio file
├── sample_2.wav                    # Sample evaluation audio file
├── sample_3.wav                    # Sample evaluation audio file
└── README.md                       # Project documentation
```

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/Shamith-247/MULTIMODAL-AI-BASED-EMOTIONAL-WELLBEING-MONITORING-SYSTEM-FOR-STUDENTS.git
cd MULTIMODAL-AI-BASED-EMOTIONAL-WELLBEING-MONITORING-SYSTEM-FOR-STUDENTS/voice_model
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Training & Evaluation
To train the full pipeline across datasets:
```bash
python SIGNALWELL_VOICE_PIPELINE.py
```
*(Or upload and run `SIGNALWELL_VOICE_COLAB.ipynb` directly on Google Colab with a T4 GPU).*

### 2. Inference on New Audio Files
```python
from SIGNALWELL_VOICE_PIPELINE import predict_single_file

predict_single_file("path/to/audio.wav")
```

---

## 📦 Dependencies

* Python 3.10+
* PyTorch >= 2.0.0
* Transformers >= 4.30.0
* SoundFile
* Librosa
* Scikit-Learn
* KaggleHub
