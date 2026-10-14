# ============================================================
# FIX — lighting-corrected inference (attacks the backlight/shadow domain gap directly)
# ============================================================
import cv2
import numpy as np
from PIL import Image

def apply_clahe(pil_img):
    """Evens out harsh directional lighting (like your backlit window) before classification."""
    img_np = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l)
    corrected_lab = cv2.merge([l_corrected, a, b])
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(corrected_rgb)

def predict_emotions_corrected(image_path_or_pil, show=True):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    with torch.no_grad():
        faces = mtcnn_infer(img)
    if faces is None:
        print("No face detected.")
        return []
    if faces.dim() == 3:
        faces = faces.unsqueeze(0)

    results = []
    model.eval()
    with torch.no_grad():
        for i in range(faces.shape[0]):
            face_raw = faces[i].clamp(0, 255).byte()
            face_pil = T.ToPILImage()(face_raw)

            # ---- the actual fix: lighting-correct the crop before classifying ----
            face_pil_corrected = apply_clahe(face_pil)

            x = eval_tf(face_pil_corrected).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = int(probs.argmax())
            results.append({
                "face_index": i,
                "top_emotion": label_names[top_idx],
                "confidence": float(probs[top_idx]),
                "all_probabilities": dict(zip(label_names, probs.tolist())),
                "face_image_original": face_pil,
                "face_image_corrected": face_pil_corrected,
            })

    if show:
        fig, axes = plt.subplots(2, len(results), figsize=(4 * len(results), 8))
        if len(results) == 1:
            axes = axes.reshape(2, 1)
        for i, r in enumerate(results):
            axes[0, i].imshow(r["face_image_original"])
            axes[0, i].set_title("Original crop")
            axes[0, i].axis("off")
            axes[1, i].imshow(r["face_image_corrected"])
            axes[1, i].set_title(f'Corrected -> {r["top_emotion"]} ({r["confidence"]*100:.1f}%)')
            axes[1, i].axis("off")
        plt.tight_layout()
        plt.show()

    return results

from google.colab import files
print("Upload a photo to test with lighting correction...")
uploaded = files.upload()
fname = list(uploaded.keys())[0]
results = predict_emotions_corrected(fname)
for r in results:
    print(f"\nFace {r['face_index']}: {r['top_emotion']} ({r['confidence']*100:.1f}%)")
    for emotion, prob in sorted(r['all_probabilities'].items(), key=lambda x: -x[1]):
        print(f"    {emotion:<10} {prob:.3f}")
