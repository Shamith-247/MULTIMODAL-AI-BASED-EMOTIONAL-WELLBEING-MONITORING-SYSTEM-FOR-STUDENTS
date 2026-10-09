# Voice Emotion Recognition (SER) Module

This directory contains the speech emotion recognition pipeline for the multimodal student wellbeing project. It takes a raw audio file and predicts one of 7 emotion categories (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) along with an uncertainty score indicating how ambiguous the speech is.

## How It Works

Instead of relying only on deep audio embeddings or only on classical audio features, the model uses a dual-branch setup:

1. **Self-Supervised Speech Representations (WavLM Large)**: The 6-second audio clip is split into six 1-second chunks and passed through pretrained WavLM Large to extract 1024-dimensional temporal embeddings.
2. **Acoustic Prosody (37 features)**: Classical speech features extracted with Librosa:
   - Pitch (F0 via YIN algorithm: mean, standard deviation, range)
   - Energy (RMS energy: mean, standard deviation, range)
   - Spectral descriptors (centroid, bandwidth, rolloff, zero-crossing rate)
   - Voice activity (pause ratio based on energy thresholding)
   - 13 MFCC means and 13 MFCC standard deviations
3. **Temporal Transformer**: Both feature sets are projected into a shared 256-dimensional space and fused, then passed through a 3-layer Transformer encoder (`heads=8`, `dim_feedforward=1024`, `dropout=0.15`).
4. **Attention Pooling**: A learned attention layer aggregates the 6 temporal chunks into a single representation based on which segments contain the strongest emotional cues.
5. **Output Heads**:
   - Emotion classification head (softmax over 7 classes)
   - Uncertainty head (sigmoid score between 0.0 and 1.0 reflecting confidence)

## Benchmark Results

The model was evaluated using a strict speaker-independent split (speakers in the test set never appeared in training or validation) across four speech emotion datasets: RAVDESS, CREMA-D, TESS, and SAVEE.

| Metric | Score |
|---|---|
| Test Accuracy | 68.20% |
| Test Macro F1 | 68.32% |
| Test UAR (Unweighted Average Recall) | 67.49% |
| Validation Macro F1 | 66.06% |

## Supported Datasets

The pipeline supports both clean acted speech and spontaneous conversational speech:

| Dataset | Type | Recordings | Role |
|---|---|---|---|
| **RAVDESS** | Acted studio | 1,440 | Clean emotional baseline |
| **CREMA-D** | Acted multi-speaker | 7,442 | Speaker demographic & ethnicity diversity |
| **TESS** | Acted female | 2,800 | High-pitch emotion clarity |
| **SAVEE** | Acted British | 480 | Dialectal variation |
| **MELD** | Conversational | ~13,700 | Spontaneous dialogue from Friends TV series |
| **ESD** | Expressive speech | ~29,000 | Multi-speaker acoustic dynamics (English & Mandarin) |
| **EmoV-DB** | Expressive | ~7,000 | Vocal nuances (amusement, sleepiness, anger) |
| **IEMOCAP** | Dyadic conversational | ~10,000 | Academic gold-standard benchmark |

You can toggle which datasets to include directly via `DATASET_CONFIG` in the script. Extracted features are automatically cached to disk (or your Google Drive in Colab) so you never lose feature extraction progress if a session restarts.

## Files in this Directory

- `SIGNALWELL_VOICE_PIPELINE.py`: Full standalone script that handles dataset downloading, feature extraction, training with early stopping, evaluation metrics, and single-file prediction.
- `SIGNALWELL_VOICE_KAGGLE.ipynb`: Master notebook configured for Kaggle Free GPU (T4 x 2, 30 hrs/week) with native Kaggle dataset access.
- `SIGNALWELL_VOICE_COLAB.ipynb`: Self-contained Google Colab notebook with Google Drive caching.
- `requirements.txt`: Python packages needed to run the code.
- `sample_1.wav`, `sample_2.wav`, `sample_3.wav`: Test audio samples to quickly verify inference.

## Getting Started

### Option 1: Kaggle Free GPU (Recommended — 30 Hours Free / Week)
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right sidebar under **Notebook Options**:
   - **Accelerator**: select **GPU T4 x 2** (or P100).
   - **Internet**: toggle **ON**.
3. Click **File -> Import Notebook** and select `SIGNALWELL_VOICE_KAGGLE.ipynb`.
4. Click **Run All**!

### Option 2: Google Colab
Upload `SIGNALWELL_VOICE_COLAB.ipynb` to Google Colab, enable GPU runtime (`Runtime -> Change runtime type -> T4 GPU`), and run the cells.

### Option 3: Running Locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/Shamith-247/MULTIMODAL-AI-BASED-EMOTIONAL-WELLBEING-MONITORING-SYSTEM-FOR-STUDENTS.git
cd MULTIMODAL-AI-BASED-EMOTIONAL-WELLBEING-MONITORING-SYSTEM-FOR-STUDENTS/voice_model
pip install -r requirements.txt
```

To run training:
```bash
python SIGNALWELL_VOICE_PIPELINE.py
```

To run prediction on a single audio file:
```python
from SIGNALWELL_VOICE_PIPELINE import predict_single_file

label, confidence = predict_single_file("sample_1.wav")
```
