# ============================================================
# QUICK TEST — sanity-check all three trained models
# ============================================================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch.nn.functional as F

BASE = "/content/drive/MyDrive/signalwell-models"

print("=== TEXT ===")
tok = AutoTokenizer.from_pretrained(f"{BASE}/text_model")
tmodel = AutoModelForSequenceClassification.from_pretrained(f"{BASE}/text_model").eval()
sample_text = "I can't believe how great today turned out, everything just worked."
inputs = tok(sample_text, return_tensors="pt")
with torch.no_grad():
    probs = F.softmax(tmodel(**inputs).logits, dim=-1)[0]
for i, p in enumerate(probs):
    print(f"  {tmodel.config.id2label[i]}: {p.item():.3f}")

print("\n=== VOICE ===")
import librosa
vfeat = Wav2Vec2FeatureExtractor.from_pretrained(f"{BASE}/voice_model_v3")
vmodel = Wav2Vec2ForSequenceClassification.from_pretrained(f"{BASE}/voice_model_v3").eval()
sample_wav_path = wav_paths[test_idx[0]]  # reuse one held-out test clip if this session still has it in memory
wav, _ = librosa.load(sample_wav_path, sr=16000, mono=True)
feats = vfeat(wav[:64000], sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    probs = F.softmax(vmodel(**feats).logits, dim=-1)[0]
for i, p in enumerate(probs):
    print(f"  {vmodel.config.id2label[i]}: {p.item():.3f}")

print("\n=== FACE ===")
# Assumes face_model / label_names_face / face_test_ds are still in memory from training this session.
# If not, reload the checkpoint the same way as text/voice above using torch.load(f"{BASE}/face_model_vggface2_rafdb_best.pt").
sample_img, sample_label = face_test_ds[0]
with torch.no_grad():
    probs = F.softmax(face_model(sample_img.unsqueeze(0).to(device)), dim=-1)[0]
for i, p in enumerate(probs):
    print(f"  {label_names_face[i]}: {p.item():.3f}")
print("  true label:", label_names_face[sample_label])
