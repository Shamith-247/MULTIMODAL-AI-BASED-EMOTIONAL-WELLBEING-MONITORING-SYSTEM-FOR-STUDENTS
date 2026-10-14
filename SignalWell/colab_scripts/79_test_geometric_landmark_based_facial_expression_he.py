# ============================================================
# TEST — geometric/landmark-based facial expression heuristic (Colab version)
# Uses MediaPipe Face Landmarker blendshapes, implementing the paper's
# described Plutchik-based associations (mouth->happy/angry/disgust,
# eyes->fear/sad, eyes+mouth+brows widening->surprise)
# ============================================================
!pip install -q mediapipe

import urllib.request, os
MODEL_PATH = "/content/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("Downloaded face_landmarker.task")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def get_blendshapes(pil_img):
    img_np = np.array(pil_img.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result = detector.detect(mp_image)
    if not result.face_blendshapes:
        return None
    return {b.category_name: b.score for b in result.face_blendshapes[0]}

def score_emotions(bs):
    """Heuristic combining blendshapes per the paper's stated Plutchik associations."""
    def g(name): return bs.get(name, 0.0)

    mouth_smile = (g('mouthSmileLeft') + g('mouthSmileRight')) / 2
    mouth_frown = (g('mouthFrownLeft') + g('mouthFrownRight')) / 2
    jaw_open = g('jawOpen')
    mouth_wide = mouth_smile + mouth_frown + jaw_open * 0.5  # paper: wide mouth -> happy/angry/disgust

    eye_wide = (g('eyeWideLeft') + g('eyeWideRight')) / 2
    eye_squint = (g('eyeSquintLeft') + g('eyeSquintRight')) / 2
    eye_blink = (g('eyeBlinkLeft') + g('eyeBlinkRight')) / 2

    brow_up = g('browInnerUp')
    brow_down = (g('browDownLeft') + g('browDownRight')) / 2
    nose_sneer = (g('noseSneerLeft') + g('noseSneerRight')) / 2

    scores = {
        'happiness': mouth_smile * 1.5 + eye_squint * 0.2,
        'surprise':  eye_wide * 1.2 + brow_up * 1.2 + jaw_open * 0.8,
        'anger':     brow_down * 1.3 + mouth_frown * 0.6 + eye_squint * 0.4,
        'disgust':   nose_sneer * 1.5 + mouth_frown * 0.4,
        'fear':      brow_up * 0.8 + eye_wide * 0.8 + jaw_open * 0.3,
        'sadness':   mouth_frown * 0.8 + brow_up * 0.4 + eye_blink * 0.3,
        'neutral':   max(0.05, 1 - (mouth_wide + eye_wide + eye_squint + brow_up + brow_down + nose_sneer)),
    }
    total = sum(scores.values()) or 1
    ranked = sorted(((k, v/total) for k, v in scores.items()), key=lambda x: -x[1])
    return ranked

from google.colab import files
print("Upload a face photo...")
uploaded = files.upload()
fname = list(uploaded.keys())[0]
img = Image.open(fname)

plt.figure(figsize=(3,3))
plt.imshow(img)
plt.axis("off")
plt.show()

bs = get_blendshapes(img)
if bs is None:
    print("No face detected.")
else:
    print("Raw blendshape values (the actual geometric measurements):")
    for k, v in sorted(bs.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k:<20} {v:.3f}")

    print("\nHeuristic emotion ranking (implements the paper's stated associations):")
    ranked = score_emotions(bs)
    for name, prob in ranked:
        bar = "█" * int(prob * 30)
        print(f"    {name:<10} {prob:.3f}  {bar}")
