# ============================================================
# DIAGNOSTIC + FIX — logit bias correction for the neutral over-prediction
# ============================================================
from google.colab import files
import torch.nn.functional as F

NEUTRAL_IDX = label_names.index('neutral')  # from your v7 session

def predict_with_bias_correction(image_path_or_pil, neutral_penalty=0.0, show=True):
    """Same as predict_emotions(), but lets you subtract a penalty from the
    neutral logit before softmax, to test how much it's actually 'winning by.'"""
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
            x = eval_tf(face_pil).unsqueeze(0).to(device)
            logits = model(x).clone()
            logits[0, NEUTRAL_IDX] -= neutral_penalty  # the actual correction
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = int(probs.argmax())
            results.append({
                "top_emotion": label_names[top_idx],
                "confidence": float(probs[top_idx]),
                "all_probabilities": dict(zip(label_names, probs.tolist())),
                "face_image": face_pil,
            })

    if show:
        for r in results:
            plt.figure(figsize=(3,3))
            plt.imshow(r["face_image"])
            plt.title(f'{r["top_emotion"]} ({r["confidence"]*100:.1f}%)')
            plt.axis("off")
            plt.show()
    return results

print("Upload the SAME photo that gave you a neutral result before...")
uploaded = files.upload()
fname = list(uploaded.keys())[0]

print("\n=== No correction (baseline) ===")
predict_with_bias_correction(fname, neutral_penalty=0.0)

print("\n=== With increasing neutral penalty ===")
for penalty in [0.5, 1.0, 1.5, 2.0, 2.5]:
    print(f"\n-- penalty = {penalty} --")
    r = predict_with_bias_correction(fname, neutral_penalty=penalty, show=False)[0]
    for emotion, prob in sorted(r['all_probabilities'].items(), key=lambda x: -x[1])[:3]:
        print(f"    {emotion:<10} {prob:.3f}")
