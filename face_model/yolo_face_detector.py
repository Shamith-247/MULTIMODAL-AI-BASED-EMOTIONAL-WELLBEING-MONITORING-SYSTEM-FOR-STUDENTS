#!/usr/bin/env python3
"""
SignalWell — Lightweight Dedicated YOLO Face Detector
======================================================
Dedicated face detector utilizing YOLO11n-face (nano architecture) from:
https://github.com/akanametov/yolo-face

Features:
- Single-pass multi-face detection with high recall under head tilt, yaw, pitch, and varying lighting.
- Accurate bounding-box extraction [x1, y1, x2, y2] and [x, y, w, h].
- Configurable and dynamically adjustable confidence threshold (default: 0.35).
- Non-Maximum Suppression (NMS IoU: 0.45) to prevent duplicate boxes.
- Safe coordinate validation and boundary clamping.
- Robust face-crop extraction with context margin for visual verification.
- Zero dependency on restrictive center-guide positions.
- CPU & CUDA GPU acceleration support.
"""

import os
import sys
import time
import urllib.request
import cv2
import numpy as np

# Primary dedicated face model release assets (akanametov/yolo-face v1.0.0)
YOLO_FACE_MODELS = [
    ("yolov11n-face.pt", "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"),
    ("yolov11n-face.onnx", "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.onnx"),
    ("yolov8n-face.pt", "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"),
    ("yolov8n-face.onnx", "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx")
]

class YOLOFaceDetector:
    """
    Dedicated lightweight YOLO face detector supporting YOLO11n-face / YOLOv8n-face.
    """
    def __init__(self, model_path=None, conf_thresh=0.35, iou_thresh=0.45, imgsz=640, device=None):
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.imgsz = imgsz
        self.model = None
        self.model_name = ""
        self.backend = None
        self.last_det_time = 0.0

        # Resolve device: CUDA GPU when available, CPU otherwise
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        # 1. Determine model candidate path
        valid_path = None
        if model_path and os.path.exists(model_path):
            valid_path = model_path
        else:
            # Check local files in priority order
            for filename, _ in YOLO_FACE_MODELS:
                local_candidate = os.path.join(models_dir, filename)
                if os.path.exists(local_candidate) and os.path.getsize(local_candidate) > 1024 * 1024:
                    valid_path = local_candidate
                    break

        # 2. Download primary model if no valid local checkpoint is found
        if valid_path is None:
            primary_name, primary_url = YOLO_FACE_MODELS[0]
            target_file = os.path.join(models_dir, primary_name)
            print(f"[YOLOFaceDetector] Downloading {primary_name} from {primary_url}...")
            try:
                # Add browser User-Agent for GitHub releases
                req = urllib.request.Request(
                    primary_url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as resp, open(target_file, 'wb') as out_f:
                    out_f.write(resp.read())
                if os.path.exists(target_file) and os.path.getsize(target_file) > 1024 * 1024:
                    valid_path = target_file
                    print(f"[YOLOFaceDetector] Successfully downloaded {primary_name} ({os.path.getsize(target_file)/1024/1024:.2f} MB)")
            except Exception as e:
                print(f"[YOLOFaceDetector] Download failed: {e}")

        # 3. Load model via Ultralytics
        if valid_path and os.path.exists(valid_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(valid_path)
                self.model_name = os.path.basename(valid_path)
                self.backend = "ultralytics"
                print(f"[YOLOFaceDetector] Loaded dedicated face model '{self.model_name}' on {self.device} (conf_thresh={self.conf_thresh})")
            except Exception as e:
                print(f"[YOLOFaceDetector] Failed to load with Ultralytics: {e}")
                self.model = None

        # 4. Fallback if Ultralytics fails
        if self.model is None:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(haar_path):
                self.model = cv2.CascadeClassifier(haar_path)
                self.model_name = "haarcascade_frontalface_default.xml"
                self.backend = "haar_fallback"
                print(f"[YOLOFaceDetector] Warning: Falling back to OpenCV Haar Cascade from {haar_path}")
            else:
                print(f"[YOLOFaceDetector] ERROR: No face detector could be initialized.")

    def set_confidence_threshold(self, new_thresh: float):
        """Allows dynamically updating detection confidence at runtime."""
        self.conf_thresh = max(0.05, min(0.95, float(new_thresh)))

    def detect(self, frame_bgr):
        """
        Detects faces in frame_bgr.
        Returns:
            faces: list of dicts:
                {
                    'bbox': (x1, y1, fw, fh),
                    'xyxy': (x1, y1, x2, y2),
                    'conf': float,
                    'center': (cx, cy)
                }
            det_time_ms: float detection latency in milliseconds
        """
        if frame_bgr is None or frame_bgr.size == 0 or self.model is None:
            return [], 0.0

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        faces = []

        if self.backend == "ultralytics":
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # Safe clamp to frame boundaries
                    x1 = max(0, min(int(x1), w - 1))
                    y1 = max(0, min(int(y1), h - 1))
                    x2 = max(0, min(int(x2), w))
                    y2 = max(0, min(int(y2), h))

                    fw = max(0, x2 - x1)
                    fh = max(0, y2 - y1)

                    # Reject degenerate bounding boxes (< 15x15 pixels)
                    if fw < 15 or fh < 15:
                        continue

                    conf = float(box.conf[0].cpu().numpy())
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    faces.append({
                        "bbox": (x1, y1, fw, fh),
                        "xyxy": (x1, y1, x2, y2),
                        "conf": conf,
                        "center": (cx, cy)
                    })

        elif self.backend == "haar_fallback":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detections = self.model.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, fw, fh) in detections:
                x1, y1, x2, y2 = x, y, x + fw, y + fh
                cx, cy = int(x + fw / 2), int(y + fh / 2)
                faces.append({
                    "bbox": (x, y, fw, fh),
                    "xyxy": (x1, y1, x2, y2),
                    "conf": 0.85,
                    "center": (cx, cy)
                })

        t1 = time.perf_counter()
        det_time_ms = (t1 - t0) * 1000.0
        self.last_det_time = det_time_ms
        return faces, det_time_ms

    def extract_face_crop(self, frame_bgr, xyxy, padding=0.12, make_square=True):
        """
        Extracts a high-fidelity face crop from frame_bgr with context margin.
        When make_square=True, calculates a 1:1 square bounding box around the face center,
        preventing aspect-ratio stretching/compression when resized for CNN inference.
        If the square box extends outside the frame boundaries, uses mirror/border padding.

        Returns:
            face_crop (numpy.ndarray or None): cropped face image
            crop_coords (tuple): (cx1, cy1, cx2, cy2) clamped coordinates
        """
        if frame_bgr is None or frame_bgr.size == 0 or xyxy is None:
            return None, (0, 0, 0, 0)

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = xyxy

        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if make_square:
            # Side length based on maximum dimension plus context margin
            side = int(max(fw, fh) * (1.0 + 2.0 * padding))
            half_side = side // 2

            box_x1 = cx - half_side
            box_y1 = cy - half_side
            box_x2 = box_x1 + side
            box_y2 = box_y1 + side

            # Determine padding needed if box extends beyond frame edges
            pad_left = max(0, -box_x1)
            pad_top = max(0, -box_y1)
            pad_right = max(0, box_x2 - w)
            pad_bottom = max(0, box_y2 - h)

            # Intersection with valid frame
            src_x1 = max(0, box_x1)
            src_y1 = max(0, box_y1)
            src_x2 = min(w, box_x2)
            src_y2 = min(h, box_y2)

            if src_x2 <= src_x1 or src_y2 <= src_y1:
                return None, (0, 0, 0, 0)

            crop_valid = frame_bgr[src_y1:src_y2, src_x1:src_x2]

            # If boundary exceeded, apply copyMakeBorder to preserve perfect square aspect ratio
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                crop = cv2.copyMakeBorder(
                    crop_valid,
                    top=pad_top,
                    bottom=pad_bottom,
                    left=pad_left,
                    right=pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )
            else:
                crop = crop_valid

            return crop, (src_x1, src_y1, src_x2, src_y2)
        else:
            pad_x = int(fw * padding)
            pad_y = int(fh * padding)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            if cx2 <= cx1 or cy2 <= cy1:
                return None, (0, 0, 0, 0)

            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                return None, (0, 0, 0, 0)

            return crop, (cx1, cy1, cx2, cy2)

if __name__ == "__main__":
    detector = YOLOFaceDetector(conf_thresh=0.35)
    test_img_path = "SignalWell/test_photos/WIN_20260901_17_11_36_Pro.jpg"
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        faces, latency = detector.detect(img)
        print(f"Self-test on {test_img_path}:")
        print(f"  Faces found: {len(faces)}")
        print(f"  Latency: {latency:.2f} ms")
        if faces:
            crop, coords = detector.extract_face_crop(img, faces[0]['xyxy'])
            print(f"  Face #1: xyxy={faces[0]['xyxy']} conf={faces[0]['conf']:.2f}")
            print(f"  Extracted crop: shape={crop.shape} coords={coords}")
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        faces, latency = detector.detect(dummy)
        print(f"Self-test: Backend={detector.backend}, Detected={len(faces)}, Latency={latency:.2f} ms")
