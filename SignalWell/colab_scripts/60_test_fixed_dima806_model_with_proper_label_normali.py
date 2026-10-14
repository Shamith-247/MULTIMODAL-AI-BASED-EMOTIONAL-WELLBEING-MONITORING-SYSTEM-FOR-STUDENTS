# ============================================================
# TEST (FIXED) — dima806 model, with proper label normalization
# ============================================================
!pip install -q transformers pillow datasets

from transformers import pipeline
from datasets import load_dataset
import random

classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

face_ds_full = load_dataset("deanngkl/raf-db-7emotions")
label_names_face = face_ds_full["train"].features["label"].names
print("RAF-DB labels:", label_names_face)

# Map dima806's short label names to RAF-DB's full label names
LABEL_NORMALIZE = {
    "happy": "happiness", "angry": "anger", "sad": "sadness",
    "disgust": "disgust", "fear": "fear", "neutral": "neutral", "surprise": "surprise",
}

ds = face_ds_full["train"]
random.seed(42)
sample_indices = random.sample(range(len(ds)), 30)

correct = 0
results_log = []
for idx in sample_indices:
    item = ds[idx]
    img = item["image"].convert("RGB")
    true_label = label_names_face[item["label"]]

    result = classifier(img)[0]
    pred_label_raw = result["label"].lower()
    pred_label = LABEL_NORMALIZE.get(pred_label_raw, pred_label_raw)
    conf = result["score"]

    is_correct = pred_label == true_label
    correct += is_correct
    marker = "✅" if is_correct else "❌"
    print(f"{marker} true={true_label:<10} pred={pred_label_raw:<10} (normalized: {pred_label:<10}) conf={conf:.3f}")
    results_log.append((true_label, pred_label, conf, is_correct))

print(f"\nCorrected accuracy on 30 samples: {correct}/30 = {correct/30*100:.1f}%")

print("\nPer-class breakdown:")
for label in label_names_face:
    class_results = [r for r in results_log if r[0] == label]
    if class_results:
        class_correct = sum(r[3] for r in class_results)
        print(f"  {label}: {class_correct}/{len(class_results)}")
