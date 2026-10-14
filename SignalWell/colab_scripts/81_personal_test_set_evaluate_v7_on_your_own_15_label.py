# ============================================================
# PERSONAL TEST SET — evaluate v7 on your own 15 labeled photos
# (5 emotions x 3 examples), with real accuracy + confusion breakdown
# ============================================================
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install facenet-pytorch --no-deps -q

from PIL import Image
print("Pillow ready")

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Load your trained v7 model ----
CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v7_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names = ckpt["labels"]
print("Model labels:", label_names)

class FaceEmotionModel(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=None, classify=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        emb = self.backbone(x)
        emb = self.dropout(emb)
        return self.fc(emb)

model = FaceEmotionModel(len(label_names)).to(device)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

mtcnn_infer = MTCNN(image_size=160, margin=20, post_process=False, keep_all=False, device=device)
eval_tf = T.Compose([T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

def predict(pil_img):
    with torch.no_grad():
        face_tensor = mtcnn_infer(pil_img)
    if face_tensor is None:
        return None, None
    face_raw = face_tensor.clamp(0, 255).byte()
    face_pil = T.ToPILImage()(face_raw)
    with torch.no_grad():
        x = eval_tf(face_pil).unsqueeze(0).to(device)
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    return probs, face_pil

# ---- Upload all 15 photos, one at a time, labeling each ----
from google.colab import files

VALID_LABELS = label_names  # anger, disgust, fear, happiness, neutral, sadness, surprise
# your set uses "angry"/"sad" informally — map those to the model's actual label spelling
LABEL_ALIASES = {"angry":"anger", "sad":"sadness", "surprised":"surprise", "disgusted":"disgust"}

results_log = []
print(f"Upload your 15 photos one at a time. After each upload, type the true emotion.")
print(f"Valid labels: {', '.join(VALID_LABELS)}  (angry/sad/surprised/disgusted also accepted)\n")

num_photos = int(input("How many photos are you uploading? "))
for i in range(num_photos):
    print(f"\n--- Photo {i+1}/{num_photos} ---")
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    img = Image.open(fname).convert("RGB")

    plt.figure(figsize=(2.5,2.5))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    while True:
        true_label = input(f"True emotion for {fname}: ").strip().lower()
        true_label = LABEL_ALIASES.get(true_label, true_label)
        if true_label in VALID_LABELS:
            break
        print(f"Not recognized. Use one of: {', '.join(VALID_LABELS)}")

    probs, face_crop = predict(img)
    if probs is None:
        print("No face detected — skipping this one.")
        continue

    pred_idx = int(probs.argmax())
    pred_label = label_names[pred_idx]
    confidence = float(probs[pred_idx])
    correct = pred_label == true_label

    marker = "✅" if correct else "❌"
    print(f"{marker} True: {true_label}  |  Predicted: {pred_label} ({confidence*100:.1f}%)")

    results_log.append({
        "true": true_label, "pred": pred_label, "conf": confidence, "correct": correct
    })

# ---- Final report ----
print("\n" + "="*50)
print("PERSONAL TEST SET RESULTS")
print("="*50)

n = len(results_log)
correct_n = sum(r["correct"] for r in results_log)
print(f"\nOverall accuracy on your photos: {correct_n}/{n} = {correct_n/n*100:.1f}%")

print("\nPer-emotion breakdown:")
for label in VALID_LABELS:
    subset = [r for r in results_log if r["true"] == label]
    if subset:
        acc = sum(r["correct"] for r in subset) / len(subset)
        print(f"  {label:<10} {sum(r['correct'] for r in subset)}/{len(subset)} = {acc*100:.0f}%")

print("\nConfusion detail (what it actually predicted when wrong):")
for r in results_log:
    if not r["correct"]:
        print(f"  true={r['true']:<10} -> predicted={r['pred']:<10} ({r['conf']*100:.0f}% confidence)")
