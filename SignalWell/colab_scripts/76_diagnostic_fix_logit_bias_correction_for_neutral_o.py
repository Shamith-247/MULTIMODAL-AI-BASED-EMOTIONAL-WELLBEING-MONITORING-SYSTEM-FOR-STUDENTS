# ============================================================
# DIAGNOSTIC + FIX — logit bias correction for neutral over-prediction
# Self-contained: loads v7 fresh from Drive first
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Load your trained v7 model ----
CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v7_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names = ckpt["labels"]
print("Labels:", label_names)
print("Reported test_acc:", ckpt.get("test_acc"))

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
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("Ignored unexpected keys (harmless, unused VGGFace2 identity head):", unexpected)
print("Missing keys (should be empty):", missing)
model.eval()
print("Model loaded.\n")

# ---- Same preprocessing as v7 training: single normalization after a raw MTCNN crop ----
mtcnn_infer = MTCNN(image_size=160, margin=20, post_process=False, keep_all=False, device=device)
eval_tf = T.Compose([T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

NEUTRAL_IDX = label_names.index('neutral')

def predict_with_bias_correction(image_path_or_pil, neutral_penalty=0.0, show=True):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    with torch.no_grad():
        face_tensor = mtcnn_infer(img)
    if face_tensor is None:
        print("No face detected.")
        return []

    face_raw = face_tensor.clamp(0, 255).byte()
    face_pil = T.ToPILImage()(face_raw)

    with torch.no_grad():
        x = eval_tf(face_pil).unsqueeze(0).to(device)
        logits = model(x).clone()
        logits[0, NEUTRAL_IDX] -= neutral_penalty
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    result = {
        "top_emotion": label_names[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probabilities": dict(zip(label_names, probs.tolist())),
        "face_image": face_pil,
    }
    if show:
        plt.figure(figsize=(3,3))
        plt.imshow(face_pil)
        plt.title(f'{result["top_emotion"]} ({result["confidence"]*100:.1f}%)')
        plt.axis("off")
        plt.show()
    return [result]

# ---- Run the sweep ----
from google.colab import files
print("Upload the SAME photo that gave you a stuck neutral result...")
uploaded = files.upload()
fname = list(uploaded.keys())[0]

print("\n=== No correction (baseline) ===")
predict_with_bias_correction(fname, neutral_penalty=0.0)

print("\n=== With increasing neutral penalty ===")
for penalty in [0.5, 1.0, 1.5, 2.0, 2.5]:
    print(f"\n-- penalty = {penalty} --")
    r = predict_with_bias_correction(fname, neutral_penalty=penalty, show=False)[0]
    for emotion, prob in sorted(r['all_probabilities'].items(), key=lambda x: -x[1])[:3]:
        print(f"    {emotion:<10} {prob:.3f}")
