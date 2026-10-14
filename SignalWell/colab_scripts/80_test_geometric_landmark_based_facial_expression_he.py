# ============================================================
# TEST — geometric/landmark-based facial expression heuristic (OpenCV version)
# Avoids MediaPipe entirely (no Python 3.13 wheels exist for it yet)
# ============================================================
!pip install -q opencv-contrib-python

import cv2, os, urllib.request
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---- Download the standard 68-point LBF landmark model (one-time, ~9MB) ----
LBF_PATH = "/content/lbfmodel.yaml"
if not os.path.exists(LBF_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
        LBF_PATH
    )
    print("Downloaded lbfmodel.yaml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_PATH)

def get_landmarks(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    ok, landmarks = facemark.fit(gray, np.array([faces[0]]))
    if not ok:
        return None, None
    return landmarks[0][0], (x, y, w, h)  # 68x2 array of (x,y) points

def dist(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def score_emotions(pts):
    """Same iBUG 68-point layout as dlib: mouth 48-67, eyes 36-47, brows 17-26, jaw 0-16."""
    mouth_width = dist(pts[48], pts[54])
    mouth_height = dist(pts[51], pts[57])
    face_width = dist(pts[0], pts[16])

    right_eye_open = (dist(pts[37],pts[41]) + dist(pts[38],pts[40])) / 2
    left_eye_open = (dist(pts[43],pts[47]) + dist(pts[44],pts[46])) / 2
    eye_open = (right_eye_open + left_eye_open) / 2

    right_brow = dist(pts[19], pts[37])
    left_brow = dist(pts[24], pts[44])
    brow_raise = (right_brow + left_brow) / 2

    mouth_mid_y = (pts[51][1] + pts[57][1]) / 2
    corner_angle = mouth_mid_y - ((pts[48][1] + pts[54][1]) / 2)  # positive = corners up (smile-ish)

    mw, mh = mouth_width/face_width, mouth_height/face_width
    eo, br, ca = eye_open/face_width, brow_raise/face_width, corner_angle/face_width

    print(f"Normalized geometry: mouth_width={mw:.3f} mouth_height={mh:.3f} "
          f"eye_open={eo:.3f} brow_raise={br:.3f} corner_angle={ca:.3f}")

    # ---- Heuristic per the paper's stated Plutchik associations ----
    mouth_wide = mw * 8 + mh * 6
    corners_up = max(0, ca) * 12
    corners_down = max(0, -ca) * 10

    scores = {
        'happiness': mouth_wide * 0.4 + corners_up,
        'surprise':  mh * 10 + br * 6,
        'anger':     mouth_wide * 0.3 + corners_down * 0.4,
        'disgust':   mouth_wide * 0.3 + corners_down * 0.3,
        'fear':      br * 5 + mh * 3,
        'sadness':   corners_down * 0.6,
        'neutral':   1.5,
    }
    total = sum(scores.values()) or 1
    return sorted(((k, v/total) for k, v in scores.items()), key=lambda x: -x[1])

from google.colab import files
print("Upload a face photo...")
uploaded = files.upload()
fname = list(uploaded.keys())[0]
img = Image.open(fname)

plt.figure(figsize=(4,4))
plt.imshow(img)
plt.axis("off")
plt.show()

pts, box = get_landmarks(img)
if pts is None:
    print("No face detected — try a clearer, front-facing photo.")
else:
    print("\nHeuristic emotion ranking:")
    for name, prob in score_emotions(pts):
        bar = "█" * int(prob * 30)
        print(f"    {name:<10} {prob:.3f}  {bar}")
