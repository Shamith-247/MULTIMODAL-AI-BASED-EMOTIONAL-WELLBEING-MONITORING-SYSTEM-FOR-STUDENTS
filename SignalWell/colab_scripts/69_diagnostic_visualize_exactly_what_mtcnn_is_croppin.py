# ============================================================
# DIAGNOSTIC — visualize exactly what MTCNN is cropping, and checksum the input
# ============================================================
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install facenet-pytorch --no-deps -q

from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
import hashlib
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v6_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names_face = ckpt["labels"]

backbone = InceptionResnetV1(pretrained=None, classify=False)
face_model = nn.Sequential(backbone, nn.Dropout(0.4), nn.Linear(512, len(label_names_face)))
face_model.load_state_dict(ckpt["state_dict"], strict=False)
face_model = face_model.to(device).eval()

mtcnn = MTCNN(image_size=160, margin=20, device=device)

def test_one_image_debug():
    print("Upload ONE image...")
    up = files.upload()
    fname = list(up.keys())[0]
    img = Image.open(fname).convert("RGB")

    img_array = np.array(img)
    print(f"\nFile: {fname}")
    print(f"Image size: {img.size}, mode: {img.mode}")
    print(f"Pixel checksum: {hashlib.md5(img_array.tobytes()).hexdigest()}")

    face_tensor = mtcnn(img)
    if face_tensor is None:
        print("MTCNN found NO face — this explains a fallback/default behavior")
        return

    print(f"Face tensor shape: {face_tensor.shape}")
    print(f"Face tensor value range: min={face_tensor.min().item():.2f}, max={face_tensor.max().item():.2f}")
    print(f"Face tensor checksum: {hashlib.md5(face_tensor.cpu().numpy().tobytes()).hexdigest()}")

    crop_display = ((face_tensor.permute(1,2,0).cpu().numpy() / 255.0)).clip(0,1)
    plt.figure(figsize=(3,3))
    plt.imshow(crop_display)
    plt.title("What MTCNN actually cropped")
    plt.axis("off")
    plt.show()

    normalized = ((face_tensor / 255.0) - 0.5) / 0.5
    normalized = normalized.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(face_model(normalized), dim=-1)[0]
    top3 = torch.topk(probs, 3)
    for score, idx in zip(top3.values, top3.indices):
        bar = "█" * int(score.item() * 30)
        print(f"    {label_names_face[idx.item()]:<10} {score.item():.3f}  {bar}")

test_one_image_debug()
