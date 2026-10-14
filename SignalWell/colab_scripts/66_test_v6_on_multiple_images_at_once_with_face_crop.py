# ============================================================
# TEST — v6 on multiple images at once, with face crop
# ============================================================
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install facenet-pytorch --no-deps -q

from PIL import Image, ImageDraw, ImageFont
print("Pillow working")

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v6_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names_face = ckpt["labels"]

backbone = InceptionResnetV1(pretrained=None, classify=False)
face_model = nn.Sequential(backbone, nn.Dropout(0.4), nn.Linear(512, len(label_names_face)))
face_model.load_state_dict(ckpt["state_dict"], strict=False)
face_model = face_model.to(device).eval()

mtcnn = MTCNN(image_size=160, margin=20, device=device)

def predict_from_raw_image(pil_img):
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None, None
    face_tensor = ((face_tensor / 255.0) - 0.5) / 0.5
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(face_model(face_tensor), dim=-1)[0]
    top_idx = probs.argmax().item()
    return label_names_face[top_idx], probs

# ---- Upload all 4 images at once (select multiple in the file picker) ----
from google.colab import files
print("Upload all 4 images together (Ctrl/Cmd+click to multi-select)...")
uploaded = files.upload()

for fname in uploaded.keys():
    img = Image.open(fname).convert("RGB")
    label, probs = predict_from_raw_image(img)
    print(f"\n{fname}")
    if label is None:
        print("  No face detected")
        continue
    top3 = torch.topk(probs, 3)
    for score, idx in zip(top3.values, top3.indices):
        bar = "█" * int(score.item() * 30)
        print(f"    {label_names_face[idx.item()]:<10} {score.item():.3f}  {bar}")
