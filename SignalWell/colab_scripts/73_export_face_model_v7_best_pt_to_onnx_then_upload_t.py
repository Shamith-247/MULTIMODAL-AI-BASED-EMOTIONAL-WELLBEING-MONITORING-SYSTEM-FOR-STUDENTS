# ============================================================
# EXPORT — face_model_v7_best.pt to ONNX, then upload to Hugging Face
# ============================================================
!pip install "pillow==11.1.0" --force-reinstall --no-cache-dir -q
!pip install facenet-pytorch --no-deps -q
!pip install -q huggingface_hub onnx onnxruntime

from PIL import Image
print("Pillow ready")

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn
from facenet_pytorch import InceptionResnetV1

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Rebuild the exact v7 architecture and load your trained weights ----
CKPT_PATH = "/content/drive/MyDrive/signalwell-models/face_model_v7_best.pt"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_names_face = ckpt["labels"]
print("Labels:", label_names_face)
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

model = FaceEmotionModel(len(label_names_face))
model.load_state_dict(ckpt["state_dict"])
model.eval()
print("Model loaded and ready for export.")

# ---- Export to ONNX ----
EXPORT_DIR = "/content/face_onnx_export"
import os
os.makedirs(EXPORT_DIR, exist_ok=True)
onnx_path = os.path.join(EXPORT_DIR, "face_model_v7.onnx")

dummy_input = torch.randn(1, 3, 160, 160)  # matches your training input shape

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=17,
)
print(f"Exported to: {onnx_path}")

# ---- Verify the export actually works before uploading ----
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(onnx_path)
test_input = np.random.randn(1, 3, 160, 160).astype(np.float32)
result = session.run(None, {"input": test_input})
print("ONNX inference test succeeded. Output shape:", result[0].shape)
assert result[0].shape == (1, len(label_names_face)), "Output shape mismatch — check the export"

# ---- Sanity check: compare ONNX output against the original PyTorch model on the same input ----
with torch.no_grad():
    torch_output = model(torch.from_numpy(test_input)).numpy()
max_diff = np.abs(torch_output - result[0]).max()
print(f"Max difference between PyTorch and ONNX outputs: {max_diff:.6f} (should be very close to 0)")

# ---- Upload to Hugging Face ----
from huggingface_hub import HfApi, create_repo, login

# ---- >>> FILL THESE IN <<< ----
HF_TOKEN = "your_hf_write_token_here"
HF_USERNAME = "Shamith247"
# --------------------------------

login(token=HF_TOKEN)

REPO_ID = f"{HF_USERNAME}/signalwell-face-emotion"
create_repo(REPO_ID, exist_ok=True)

api = HfApi()
api.upload_file(
    path_or_fileobj=onnx_path,
    path_in_repo="face_model_v7.onnx",
    repo_id=REPO_ID,
)

# Also upload the label list so it's documented alongside the model
import json
labels_path = os.path.join(EXPORT_DIR, "labels.json")
with open(labels_path, "w") as f:
    json.dump(label_names_face, f)
api.upload_file(path_or_fileobj=labels_path, path_in_repo="labels.json", repo_id=REPO_ID)

print(f"\n================ DONE ================")
print(f"Model live at: https://huggingface.co/{REPO_ID}")
print(f"Direct .onnx URL to use in the app:")
print(f"https://huggingface.co/{REPO_ID}/resolve/main/face_model_v7.onnx")
