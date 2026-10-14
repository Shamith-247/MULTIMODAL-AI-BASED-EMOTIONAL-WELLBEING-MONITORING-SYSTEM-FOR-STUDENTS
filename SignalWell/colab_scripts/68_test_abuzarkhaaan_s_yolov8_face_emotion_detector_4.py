# ============================================================
# TEST — abuzarkhaaan's YOLOv8 face-emotion detector (4 classes: angry, sad, surprised, happy)
# ============================================================
!pip install -q ultralytics

import os
os.system("git clone https://github.com/abuzarkhaaan/face_emotion_detection_using_yolov8.git")

# The weights are bundled in a .rar file inside the repo — need unrar to extract
!apt-get install -y unrar -q
os.system("unrar x -o+ /content/face_emotion_detection_using_yolov8/Face_emotion_detection_using_YOLOv8.rar /content/yolo_extracted/")

# Find the actual .pt weights file wherever it landed after extraction
import glob
pt_files = glob.glob("/content/yolo_extracted/**/*.pt", recursive=True)
print("Found weight files:", pt_files)

if not pt_files:
    print("No .pt file found — the archive structure may differ from what the README implies. "
          "Check /content/yolo_extracted/ manually before proceeding.")
else:
    from ultralytics import YOLO
    model = YOLO(pt_files[0])
    print("Model classes:", model.names)

    from google.colab import files
    print("Upload a test image...")
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]

    results = model(img_path)
    results[0].show()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: {model.names[cls_id]}  confidence={conf:.3f}")
