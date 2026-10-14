# ============================================================
# TEST — Voice emotion model
# ============================================================
from google.colab import drive, files
drive.mount('/content/drive')

!pip install -q librosa soundfile

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch, torch.nn.functional as F, librosa

# >>> Change this to whichever voice model folder you actually ended up with <<<
VOICE_MODEL_DIR = "/content/drive/MyDrive/signalwell-models/voice_model_v3"

feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(VOICE_MODEL_DIR)
voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(VOICE_MODEL_DIR).eval()

print("Upload a short .wav clip of someone speaking (a few seconds)...")
uploaded = files.upload()
wav_path = list(uploaded.keys())[0]

wav, _ = librosa.load(wav_path, sr=16000, mono=True)
wav = wav[:16000*4]  # first 4 seconds, matching training

inputs = feat_extractor(wav, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    probs = F.softmax(voice_model(**inputs).logits, dim=-1)[0]
top_idx = probs.argmax().item()
print(f"\nPredicted: {voice_model.config.id2label[top_idx]} ({probs[top_idx]*100:.1f}% confidence)")
for i, p in enumerate(probs):
    print(f"  {voice_model.config.id2label[i]}: {p.item():.3f}")
