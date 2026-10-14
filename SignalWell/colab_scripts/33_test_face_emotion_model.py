# ============================================================
# TEST — Face emotion model
# ============================================================
from google.colab import drive, files
drive.mount('/content/drive')

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from PIL import Image

CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_vggface2_rafdb_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names_face = ckpt["labels"]
print("Loaded checkpoint — reported test_acc during training:", ckpt.get("test_acc"))

backbone = InceptionResnetV1(pretrained=None, classify=False)
face_model = nn.Sequential(backbone, nn.Dropout(0.4), nn.Linear(512, len(label_names_face)))
face_model.load_state_dict(ckpt["state_dict"])
face_model.eval()

eval_tf = T.Compose([T.Resize((160,160)), T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

print("Upload a face photo to test...")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

img = Image.open(img_path).convert("RGB")
tensor = eval_tf(img).unsqueeze(0)
with torch.no_grad():
    probs = F.softmax(face_model(tensor), dim=-1)[0]
top_idx = probs.argmax().item()
print(f"\nPredicted: {label_names_face[top_idx]} ({probs[top_idx]*100:.1f}% confidence)")
for i, p in enumerate(probs):
    print(f"  {label_names_face[i]}: {p.item():.3f}")
