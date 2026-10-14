# ============================================================
# TEST — your v6 face model (RAF-DB accuracy check + single-image test with diagnostics)
# ============================================================
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install facenet-pytorch --no-deps -q

from PIL import Image
print("Pillow ready")

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from datasets import load_dataset
import random, hashlib
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- Load your trained model ----
CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v6_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names_face = ckpt["labels"]
print("Labels:", label_names_face)
print("Reported training test_acc:", ckpt.get("test_acc"))

backbone = InceptionResnetV1(pretrained=None, classify=False)
face_model = nn.Sequential(backbone, nn.Dropout(0.4), nn.Linear(512, len(label_names_face)))
face_model.load_state_dict(ckpt["state_dict"], strict=False)
face_model = face_model.to(device).eval()
print("Model loaded.\n")

mtcnn = MTCNN(image_size=160, margin=20, device=device)

def predict(pil_img):
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None, None, None
    normalized = ((face_tensor / 255.0) - 0.5) / 0.5
    with torch.no_grad():
        probs = F.softmax(face_model(normalized.unsqueeze(0).to(device)), dim=-1)[0]
    return label_names_face[probs.argmax().item()], probs, face_tensor

# ============================================================
# PART 1 — accuracy check on real RAF-DB test images
# ============================================================
print("="*60)
print("PART 1: RAF-DB accuracy check (30 samples)")
print("="*60)

import torchvision.transforms as T
eval_tf = T.Compose([T.Resize((160,160)), T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

face_ds_full = load_dataset("deanngkl/raf-db-7emotions")
ds = face_ds_full["train"]
random.seed(42)
sample_indices = random.sample(range(len(ds)), 30)

correct = 0
for idx in sample_indices:
    item = ds[idx]
    img = item["image"].convert("RGB")
    true_label = label_names_face[item["label"]]
    tensor = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(face_model(tensor), dim=-1)[0]
    pred_label = label_names_face[probs.argmax().item()]
    correct += (pred_label == true_label)

print(f"RAF-DB spot-check accuracy: {correct}/30 = {correct/30*100:.1f}%\n")

# ============================================================
# PART 2 — test with YOUR OWN uploaded photo, with full diagnostics
# ============================================================
print("="*60)
print("PART 2: your own uploaded photo (with diagnostics)")
print("="*60)

from google.colab import files
print("Upload ONE photo now...")
uploaded = files.upload()
fname = list(uploaded.keys())[0]

img = Image.open(fname).convert("RGB")
img_array = np.array(img)
print(f"\nFile: {fname}")
print(f"Image size: {img.size}")
print(f"Pixel checksum: {hashlib.md5(img_array.tobytes()).hexdigest()}")

label, probs, face_tensor = predict(img)

if label is None:
    print("MTCNN found NO face in this image.")
else:
    print(f"Face tensor checksum: {hashlib.md5(face_tensor.cpu().numpy().tobytes()).hexdigest()}")
    crop_display = ((face_tensor.permute(1,2,0).cpu().numpy() / 255.0)).clip(0,1)
    plt.figure(figsize=(3,3))
    plt.imshow(crop_display)
    plt.title("What MTCNN actually cropped")
    plt.axis("off")
    plt.show()

    top3 = torch.topk(probs, 3)
    print("\nPrediction:")
    for score, idx in zip(top3.values, top3.indices):
        bar = "█" * int(score.item() * 30)
        print(f"    {label_names_face[idx.item()]:<10} {score.item():.3f}  {bar}")
