# ============================================================
# TEST — v6 with proper face detection + cropping first (matches training distribution)
# ============================================================
from facenet_pytorch import MTCNN
from PIL import Image
import torch

mtcnn = MTCNN(image_size=160, margin=20, device=device)  # crops + resizes to exactly what v6 expects

def predict_from_raw_image(pil_img):
    face_tensor = mtcnn(pil_img)  # detects the face, crops it, resizes to 160x160
    if face_tensor is None:
        return None, "No face detected in the image"
    face_tensor = ((face_tensor / 255.0) - 0.5) / 0.5  # match the training normalization
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(face_model(face_tensor), dim=-1)[0]
    top_idx = probs.argmax().item()
    return label_names_face[top_idx], probs[top_idx].item()

# Test with an uploaded photo (or a webcam capture you save and upload)
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
