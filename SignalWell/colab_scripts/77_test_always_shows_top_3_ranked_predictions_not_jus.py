# ============================================================
# TEST — always shows top-3 ranked predictions, not just the winner
# ============================================================
def predict_top_k(image_path_or_pil, k=3, show=True):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    with torch.no_grad():
        face_tensor = mtcnn_infer(img)
    if face_tensor is None:
        print("No face detected.")
        return []

    face_raw = face_tensor.clamp(0, 255).byte()
    face_pil = T.ToPILImage()(face_raw)

    with torch.no_grad():
        x = eval_tf(face_pil).unsqueeze(0).to(device)
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()

    ranked = sorted(zip(label_names, probs), key=lambda p: -p[1])[:k]

    if show:
        plt.figure(figsize=(3,3))
        plt.imshow(face_pil)
        title = " / ".join(f"{name} {prob*100:.0f}%" for name, prob in ranked[:2])
        plt.title(title, fontsize=10)
        plt.axis("off")
        plt.show()

    print("Top predictions:")
    for name, prob in ranked:
        bar = "█" * int(prob * 30)
        print(f"    {name:<10} {prob:.3f}  {bar}")
    return ranked

from google.colab import files
print("Upload a photo...")
uploaded = files.upload()
fname = list(uploaded.keys())[0]
predict_top_k(fname, k=3)
