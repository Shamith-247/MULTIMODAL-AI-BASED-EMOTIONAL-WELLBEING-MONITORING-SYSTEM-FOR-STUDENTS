# ============================================================
# DIAGNOSTIC — visualize exactly what MTCNN is cropping, and checksum the input
# ============================================================
import hashlib
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

def test_one_image_debug():
    print("Upload ONE image...")
    up = files.upload()
    fname = list(up.keys())[0]
    img = Image.open(fname).convert("RGB")

    # Confirm this is genuinely a new/different image
    img_array = np.array(img)
    print(f"\nFile: {fname}")
    print(f"Image size: {img.size}, mode: {img.mode}")
    print(f"Pixel checksum: {hashlib.md5(img_array.tobytes()).hexdigest()}")

    # Run MTCNN and SHOW what it actually cropped
    face_tensor = mtcnn(img)
    if face_tensor is None:
        print("MTCNN found NO face — this would explain a fallback/default behavior")
        return

    print(f"Face tensor shape: {face_tensor.shape}")
    print(f"Face tensor value range: min={face_tensor.min().item():.2f}, max={face_tensor.max().item():.2f}")
    print(f"Face tensor checksum: {hashlib.md5(face_tensor.cpu().numpy().tobytes()).hexdigest()}")

    # Actually display the cropped face MTCNN produced
    crop_display = ((face_tensor.permute(1,2,0).cpu().numpy() / 255.0)).clip(0,1)
    plt.figure(figsize=(3,3))
    plt.imshow(crop_display)
    plt.title("What MTCNN actually cropped")
    plt.axis("off")
    plt.show()

    # Now run the actual prediction
    normalized = ((face_tensor / 255.0) - 0.5) / 0.5
    normalized = normalized.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(face_model(normalized), dim=-1)[0]
    top3 = torch.topk(probs, 3)
    for score, idx in zip(top3.values, top3.indices):
        bar = "█" * int(score.item() * 30)
        print(f"    {label_names_face[idx.item()]:<10} {score.item():.3f}  {bar}")

test_one_image_debug()
