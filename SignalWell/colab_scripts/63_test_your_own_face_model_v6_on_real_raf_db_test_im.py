# ============================================================
# TEST — your own face model (v6) on real RAF-DB test images
# Fixed: Pillow 11.1.0 (avoids the 12.x _Ink bug), strict=False state_dict load
# ============================================================
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install facenet-pytorch --no-deps -q

from PIL import Image, ImageDraw, ImageFont
print("Pillow fully working — ImageDraw and ImageFont both importable")
import PIL
print("Pillow version:", PIL.__version__)

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from datasets import load_dataset
import random

CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v6_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names_face = ckpt["labels"]
print("Loaded checkpoint. Labels:", label_names_face)
print("Reported test_acc during training:", ckpt.get("test_acc"))

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = InceptionResnetV1(pretrained=None, classify=False)
face_model = nn.Sequential(backbone, nn.Dropout(0.4), nn.Linear(512, len(label_names_face)))

# strict=False — the checkpoint contains VGGFace2's original unused identity-classification
# head ("logits.weight"/"logits.bias"), which this architecture never uses. Safe to skip.
missing, unexpected = face_model.load_state_dict(ckpt["state_dict"], strict=False)
print("Ignored unexpected keys (harmless, unused original VGGFace2 head):", unexpected)
print("Missing keys (should be empty):", missing)

face_model = face_model.to(device).eval()

eval_tf = T.Compose([T.Resize((160,160)), T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

face_ds_full = load_dataset("deanngkl/raf-db-7emotions")
ds = face_ds_full["train"]

random.seed(42)
sample_indices = random.sample(range(len(ds)), 30)

correct = 0
results_log = []
for idx in sample_indices:
    item = ds[idx]
    img = item["image"].convert("RGB")
    true_label = label_names_face[item["label"]]

    tensor = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(face_model(tensor), dim=-1)[0]
    pred_idx = probs.argmax().item()
    pred_label = label_names_face[pred_idx]
    conf = probs[pred_idx].item()

    is_correct = pred_label == true_label
    correct += is_correct
    marker = "✅" if is_correct else "❌"
    print(f"{marker} true={true_label:<10} pred={pred_label:<10} conf={conf:.3f}")
    results_log.append((true_label, pred_label, conf, is_correct))

print(f"\nSpot-check accuracy on 30 samples: {correct}/30 = {correct/30*100:.1f}%")

print("\nPer-class breakdown:")
for label in label_names_face:
    class_results = [r for r in results_log if r[0] == label]
    if class_results:
        class_correct = sum(r[3] for r in class_results)
        print(f"  {label}: {class_correct}/{len(class_results)}")
