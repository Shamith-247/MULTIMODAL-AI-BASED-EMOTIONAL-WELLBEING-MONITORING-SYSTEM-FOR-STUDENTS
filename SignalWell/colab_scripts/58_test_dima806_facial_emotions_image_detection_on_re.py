# ============================================================
# TEST — dima806/facial_emotions_image_detection on real RAF-DB test images
# ============================================================
!pip install -q transformers pillow

from transformers import pipeline
from datasets import load_dataset
import random

classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Reuse your RAF-DB test set for a fair, apples-to-apples comparison against your own v5/v6 results
if "face_ds_full" not in dir():
    from datasets import load_dataset
    face_ds_full = load_dataset("deanngkl/raf-db-7emotions")

label_names_face = face_ds_full["train"].features["label"].names
test_ds = face_ds_full["train"]  # using train split here just for a quick spot-check sample

random.seed(42)
sample_indices = random.sample(range(len(test_ds)), 20)

correct = 0
for idx in sample_indices:
    item = test_ds[idx]
    img = item["image"].convert("RGB")
    true_label = label_names_face[item["label"]]

    result = classifier(img)[0]
    pred_label = result["label"].lower()
    conf = result["score"]

    is_correct = pred_label == true_label
    correct += is_correct
    marker = "✅" if is_correct else "❌"
    print(f"{marker} true={true_label:<10} pred={pred_label:<10} conf={conf:.3f}")

print(f"\nSpot-check accuracy on 20 samples: {correct}/20 = {correct/20*100:.0f}%")
