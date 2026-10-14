# ============================================================
# COLLECT — upload your own photos, label them interactively
# ============================================================
from google.colab import files
import matplotlib.pyplot as plt
from PIL import Image
import os

LABELS_ORDER = label_names  # reuse from your v7 session: anger, disgust, fear, happiness, neutral, sadness, surprise
os.makedirs("/content/my_photos", exist_ok=True)

print("Upload as many of your own photos as you can, showing a real range of your natural expressions.")
print("Aim for at least 5-8 per emotion if possible — fear/disgust are hard to capture naturally, that's okay if those end up thin.")
uploaded = files.upload()

my_dataset = []  # list of (filepath, label)

for fname in uploaded.keys():
    filepath = f"/content/my_photos/{fname}"
    with open(filepath, "wb") as f:
        f.write(uploaded[fname])

    img = Image.open(filepath).convert("RGB")
    plt.figure(figsize=(3,3))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print(f"Label options: {', '.join(LABELS_ORDER)}")
    while True:
        label = input(f"What emotion is this ({fname})? ").strip().lower()
        if label in LABELS_ORDER:
            break
        print(f"'{label}' not recognized — type exactly one of: {', '.join(LABELS_ORDER)}")

    my_dataset.append((filepath, label))
    print(f"-> labeled as: {label}\n")

print(f"\nCollected {len(my_dataset)} labeled photos.")
from collections import Counter
print("Distribution:", Counter([l for _, l in my_dataset]))
