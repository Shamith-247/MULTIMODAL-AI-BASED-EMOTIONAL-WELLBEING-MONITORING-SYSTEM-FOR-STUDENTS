# ============================================================
# TEST — v6 with proper face detection + cropping first
# ============================================================
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install facenet-pytorch --no-deps -q

from PIL import Image, ImageDraw, ImageFont
print("Pillow working")
import PIL
print("Pillow version:", PIL.__version__)

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v6_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names_face = ckpt["labels"]
print("Loaded checkpoint. Labels:", label_names_face)

backbone = InceptionResnetV1(pretrained=None, classify=False)
face_model = nn.Sequential(backbone, nn.Dropout(0.4), nn.Linear(512, len(label_names_face)))
missing, unexpected = face_model.load_state_dict(ckpt["state_dict"], strict=False)
face_model = face_model.to(device).eval()
print("Model ready.")

mtcnn = MTCNN(image_size=160, margin=20, device=device)  # detects + crops the face to what v6 expects

def predict_from_raw_image(pil_img):
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None, "No face detected in the image"
    face_tensor = ((face_tensor / 255.0) - 0.5) / 0.5
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(face_model(face_tensor), dim=-1)[0]
    top_idx = probs.argmax().item()
    return label_names_face[top_idx], probs[top_idx].item()

from google.colab import files
print("Upload a raw, uncropped face photo (like a webcam shot)...")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = Image.open(img_path).convert("RGB")

label, conf = predict_from_raw_image(img)
if label:
    print(f"Predicted: {label} ({conf*100:.1f}% confidence)")
else:
    print(label)
