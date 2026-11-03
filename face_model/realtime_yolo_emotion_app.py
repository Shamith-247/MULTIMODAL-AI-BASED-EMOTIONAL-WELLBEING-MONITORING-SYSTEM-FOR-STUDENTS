#!/usr/bin/env python3
"""
SignalWell — Complete Real-Time YOLO Facial Emotion Recognition Application
=============================================================================
Integrates dedicated YOLO11n-face detection with the trained facial emotion
classifier checkpoint 'face_model_v6_best.pt'.

Features:
- Dedicated YOLO Face Detection: single-pass, multi-face, robust across yaw/pitch/tilt.
- Exact Model Architecture: InceptionResnetV1 (VGGFace2 backbone) + Dropout(0.4) + Linear(512, 7).
- Exact v6 Normalization & Preprocessing: 160x160 RGB, ((pixel / 255.0) - 0.5) / 0.5 -> [-1.0, 1.0].
- 7 Ekman Categories: Happy, Sad, Disgust, Surprise, Fear, Angry, Neutral.
- Strict Failure Handling (Section 6):
    * No face detected: "NO FACE DETECTED"
    * Classification failed / invalid crop: "EMOTION CLASSIFICATION UNAVAILABLE"
    * Genuine Neutral: "NEUTRAL"
- Temporal Prediction Stability: EMA smoothing (alpha=0.35) on raw probability vectors.
- Telemetry HUD: Det Latency (ms), Cls Latency (ms), Total Latency (ms), FPS, Face Count.
- Picture-in-Picture (PiP) extracted face debug inset preview.
- Cumulative Session Emotion Analytics and JSON Log Export.
- Responsive, non-blocking DirectShow threaded capture (30+ FPS).
"""

import os
import sys
import time
import json
import argparse
import threading
from collections import Counter, deque

import cv2
import numpy as np

# Ensure root paths resolve cleanly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from yolo_face_detector import YOLOFaceDetector
from emotion_classifier import (
    FacialEmotionClassifier,
    DISPLAY_CLASSES,
    EMOTION_EMOJIS
)

# Color Theme (BGR)
COLOR_BG_DARK       = (18, 22, 28)
COLOR_PANEL_BG      = (25, 30, 38)
COLOR_BORDER        = (50, 60, 75)
COLOR_TEXT_WHITE    = (240, 242, 245)
COLOR_TEXT_MUTED    = (150, 160, 175)

# Status & Emotion Colors
COLOR_EMERALD       = (154, 227, 95)    # Happy / Success
COLOR_AMBER         = (75, 168, 229)    # Searching / Warning
COLOR_ORANGE        = (22, 115, 249)    # No face
COLOR_CRIMSON       = (68, 68, 239)     # Error / Stop
COLOR_CYAN          = (248, 189, 56)    # Initializing / Info
COLOR_VIOLET        = (247, 85, 168)    # Surprise / Multi-face
COLOR_MUTED_GRAY    = (125, 135, 145)   # Neutral

EMOTION_COLORS = {
    'Happy':    (154, 227, 95),   # Emerald Green
    'Sad':      (225, 120, 85),   # Slate Blue
    'Disgust':  (85, 199, 122),   # Moss Green
    'Surprise': (247, 85, 168),   # Violet / Magenta
    'Fear':     (60, 150, 240),   # Warning Amber / Orange
    'Angry':    (68, 68, 239),    # Crimson Red
    'Neutral':  (145, 155, 165)   # Calm Gray
}

# Window Dimensions
FRAME_WIDTH  = 1140
FRAME_HEIGHT = 720

class WebcamCaptureThread:
    """
    Dedicated thread for camera polling with Windows DirectShow backend.
    """
    def __init__(self, camera_idx=0, width=1280, height=720):
        self.camera_idx = camera_idx
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.has_error = False
        self.error_msg = ""
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return True

        self.has_error = False
        self.error_msg = ""

        # DirectShow on Windows, fallback to standard VideoCapture
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap or not self.cap.isOpened():
            self.has_error = True
            self.error_msg = f"Cannot open webcam at index {self.camera_idx}"
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

class RealTimeYOLOEmotionApp:
    """
    Integrated YOLO Face Detection + Emotion Recognition Application.
    """
    def __init__(self, camera_idx=0, conf_thresh=0.35, imgsz=640):
        self.camera_idx = camera_idx
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        print("\n==================================================================")
        print(" SIGNALWELL — INTEGRATED YOLOv11 & EMOTION RECOGNITION SYSTEM")
        print("==================================================================")

        # 1. Initialize YOLO Face Detector (loaded once)
        self.detector = YOLOFaceDetector(conf_thresh=self.conf_thresh, imgsz=self.imgsz)

        # 2. Initialize Emotion Classifier (face_model_v6_best.pt loaded once)
        self.classifier = FacialEmotionClassifier(ema_alpha=0.35)

        # 3. Threaded Webcam Capture
        self.camera = WebcamCaptureThread(camera_idx=self.camera_idx)
        self.camera_active = False

        # Status Tracking
        self.det_status = "INITIALIZING CAMERA"
        self.cls_status = "READY"

        # Performance Metrics
        self.fps = 0.0
        self.det_latency_ms = 0.0
        self.cls_latency_ms = 0.0
        self.total_latency_ms = 0.0

        # Face Data & Per-Face EMA Tracking
        self.current_faces_data = []
        self.face_ema_tracker = {}
        self.primary_crop = None

        # Session Analytics Recording (Section 9)
        self.session_start_time = time.time()
        self.session_samples = []
        self.emotion_counter = Counter()

        # UI Button Coordinates (x1, y1, x2, y2)
        self.btn_camera = (20, 14, 180, 48)
        self.btn_snap   = (195, 14, 325, 48)
        self.btn_export = (340, 14, 480, 48)

        self.window_name = "SignalWell — Live YOLO Facial Emotion Recognition"

    def toggle_camera(self):
        """Toggles the live webcam feed."""
        if not self.camera_active:
            print("[Webcam] Starting built-in webcam...")
            self.det_status = "INITIALIZING CAMERA"
            if self.camera.start():
                self.camera_active = True
                self.det_status = "SEARCHING FOR FACE"
                print("[Webcam] Camera started successfully.")
            else:
                self.camera_active = False
                self.det_status = "CAMERA ERROR"
                print(f"[Webcam] Error: {self.camera.error_msg}")
        else:
            self.camera_active = False
            self.camera.stop()
            self.det_status = "NO FACE DETECTED"
            self.current_faces_data = []
            self.primary_crop = None
            print("[Webcam] Camera stopped.")

    def save_snapshot(self, frame=None):
        """Saves current annotated snapshot to disk."""
        if frame is None:
            return
        os.makedirs("snapshots", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("snapshots", f"SignalWell_Emotion_Snapshot_{ts}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"📸 Saved snapshot: {os.path.abspath(out_path)}")

    def export_analytics_log(self):
        """Exports the session analytics to a structured JSON file."""
        duration_sec = round(time.time() - self.session_start_time, 2)
        total_valid_frames = len(self.session_samples)

        distribution = {}
        for c in DISPLAY_CLASSES:
            count = self.emotion_counter[c]
            pct = round((count / total_valid_frames * 100), 1) if total_valid_frames > 0 else 0.0
            distribution[c] = {"count": count, "percentage": pct}

        dominant_emotion = self.emotion_counter.most_common(1)[0][0] if self.emotion_counter else "None"

        log_data = {
            "project": "SignalWell",
            "session_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_sec,
            "total_classified_frames": total_valid_frames,
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": distribution,
            "model_metadata": {
                "face_detector": self.detector.model_name,
                "emotion_classifier": os.path.basename(self.classifier.model_path),
                "device": self.classifier.device,
                "ema_alpha": self.classifier.ema_alpha
            },
            "timeline": self.session_samples
        }

        os.makedirs("logs", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("logs", f"SignalWell_Session_Emotion_Log_{ts}.json")
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"\n📊 Exported session analytics log ({total_valid_frames} frames) to: {os.path.abspath(filename)}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handles interactive on-screen button clicks."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Start / Stop Camera button
        bx1, by1, bx2, by2 = self.btn_camera
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            self.toggle_camera()
            return

        # Snapshot button
        sx1, sy1, sx2, sy2 = self.btn_snap
        if sx1 <= x <= sx2 and sy1 <= y <= sy2:
            self.save_snapshot()
            return

        # Export log button
        lx1, ly1, lx2, ly2 = self.btn_export
        if lx1 <= x <= lx2 and ly1 <= y <= ly2:
            self.export_analytics_log()
            return

    def draw_hud(self, frame):
        """Renders comprehensive HUD with bounding boxes, badges, bars, PiP, and analytics."""
        h, w = frame.shape[:2]

        # 1. Update Detection and Classification Status
        if not self.camera_active:
            if self.camera.has_error:
                self.det_status = "CAMERA ERROR"
                status_color = COLOR_CRIMSON
            else:
                self.det_status = "SEARCHING FOR FACE"
                status_color = COLOR_AMBER
        else:
            num_faces = len(self.current_faces_data)
            if num_faces == 0:
                self.det_status = "NO FACE DETECTED"
                status_color = COLOR_ORANGE
            elif num_faces == 1:
                self.det_status = "FACE DETECTED"
                status_color = COLOR_EMERALD
            else:
                self.det_status = "MULTIPLE FACES DETECTED"
                status_color = COLOR_VIOLET

        # 2. Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 64), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, 64), (w, 64), COLOR_BORDER, 1)

        # Camera Start/Stop Button
        bx1, by1, bx2, by2 = self.btn_camera
        btn_bg = COLOR_EMERALD if not self.camera_active else COLOR_CRIMSON
        btn_txt = "[> START CAMERA]" if not self.camera_active else "[STOP CAMERA]"
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_bg, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_TEXT_WHITE, 1)
        cv2.putText(frame, btn_txt, (bx1 + 10, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 15, 20), 2)

        # Snapshot Button
        sx1, sy1, sx2, sy2 = self.btn_snap
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), COLOR_BORDER, 1)
        cv2.putText(frame, "SNAPSHOT (S)", (sx1 + 10, sy1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_MUTED, 1)

        # Export Log Button
        lx1, ly1, lx2, ly2 = self.btn_export
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_BORDER, 1)
        cv2.putText(frame, "EXPORT LOG (L)", (lx1 + 8, ly1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_MUTED, 1)

        # Central Status Banner
        status_text = f"* {self.det_status} *"
        stat_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        stat_x = max(500, (w - stat_size[0]) // 2)
        cv2.rectangle(frame, (stat_x - 12, 12), (stat_x + stat_size[0] + 12, 52), (12, 16, 22), -1)
        cv2.rectangle(frame, (stat_x - 12, 12), (stat_x + stat_size[0] + 12, 52), status_color, 2)
        cv2.putText(frame, status_text, (stat_x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

        # Telemetry HUD (Top-right)
        telemetry = f"FPS: {self.fps:4.1f} | Det: {self.det_latency_ms:4.1f}ms | Cls: {self.cls_latency_ms:4.1f}ms | Tot: {self.total_latency_ms:4.1f}ms"
        cv2.putText(frame, telemetry, (w - 410, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_WHITE, 1)

        # 3. Standby Screen when Camera is Off
        if not self.camera_active:
            cv2.rectangle(frame, (0, 65), (w, h - 30), (12, 15, 20), -1)
            cv2.putText(frame, "WEBCAM IS PAUSED / STOPPED", (w // 2 - 200, h // 2 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 105, 125), 2)
            cv2.putText(frame, "Click [> START CAMERA] or press [SPACEBAR] to begin", (w // 2 - 240, h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MUTED, 1)
            return frame

        # 4. Render Detected Faces, Emotion Badges, and Lateral Probability Bars
        for item in self.current_faces_data:
            x1, y1, x2, y2 = item['xyxy']
            fw, fh = x2 - x1, y2 - y1
            top_em = item['top_emotion']
            top_conf = item['confidence']
            probs = item['probabilities']

            face_color = EMOTION_COLORS.get(top_em, COLOR_EMERALD) if top_em else COLOR_AMBER

            # Bounding Box with Corner Accents
            cv2.rectangle(frame, (x1, y1), (x2, y2), face_color, 2)
            corner_len = max(14, int(min(fw, fh) * 0.18))
            thick = 3
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), COLOR_TEXT_WHITE, thick)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), COLOR_TEXT_WHITE, thick)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), COLOR_TEXT_WHITE, thick)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), COLOR_TEXT_WHITE, thick)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), COLOR_TEXT_WHITE, thick)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), COLOR_TEXT_WHITE, thick)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), COLOR_TEXT_WHITE, thick)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), COLOR_TEXT_WHITE, thick)

            # Floating Emotion Badge (Centered above face)
            badge_w = max(210, fw)
            badge_h = 42
            badge_x = max(10, min(w - badge_w - 10, x1 + (fw - badge_w) // 2))
            badge_y = max(70, y1 - badge_h - 10)

            cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), COLOR_BG_DARK, -1)
            cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), face_color, 1)

            if top_em is not None:
                emoji = EMOTION_EMOJIS.get(top_em, "")
                badge_title = f"{emoji} {top_em.upper()} ({top_conf*100:.1f}%)"
                badge_sub = f"DET: {item['det_conf']*100:.1f}% • EMA SMOOTHED"
                cv2.putText(frame, badge_title, (badge_x + 10, badge_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54, COLOR_TEXT_WHITE, 2)
                cv2.putText(frame, badge_sub, (badge_x + 10, badge_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, face_color, 1)
            else:
                cv2.putText(frame, "EMOTION CLASSIFICATION UNAVAILABLE", (badge_x + 8, badge_y + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CRIMSON, 1)

            # Lateral Probability Distribution Bars (drawn to the right of face if space permits)
            if probs is not None:
                bar_x = x2 + 14 if (x2 + 175 < w - 210) else max(10, x1 - 175)
                bar_y = y1
                bar_w = 160
                bar_h = 138

                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (12, 15, 20), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_BORDER, 1)
                cv2.putText(frame, "7-EMOTION SPECTRUM", (bar_x + 8, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1)

                for r_idx, em_name in enumerate(DISPLAY_CLASSES):
                    prob_val = probs.get(em_name, 0.0)
                    ry = bar_y + 26 + (r_idx * 15)
                    # Label
                    cv2.putText(frame, em_name[:3], (bar_x + 8, ry + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_MUTED, 1)
                    # Bar track
                    cv2.rectangle(frame, (bar_x + 36, ry + 2), (bar_x + 115, ry + 9), (30, 36, 46), -1)
                    # Fill
                    fill_w = int(79 * prob_val)
                    bar_col = EMOTION_COLORS.get(em_name, COLOR_TEXT_WHITE)
                    if fill_w > 0:
                        cv2.rectangle(frame, (bar_x + 36, ry + 2), (bar_x + 36 + fill_w, ry + 9), bar_col, -1)
                    # Percentage
                    cv2.putText(frame, f"{prob_val*100:3.0f}%", (bar_x + 120, ry + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_WHITE, 1)

        # 5. Picture-in-Picture (PiP) Extracted Face Crop Preview (Top-Right)
        pip_w, pip_h = 190, 200
        pip_x = w - pip_w - 18
        pip_y = 74

        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_BG_DARK, -1)
        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_BORDER, 1)
        cv2.putText(frame, "EXTRACTED CROP (DEBUG)", (pip_x + 10, pip_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)

        if self.primary_crop is not None and self.primary_crop.size > 0:
            crop_h, crop_w = self.primary_crop.shape[:2]
            inner_w, inner_h = 170, 150
            scale = min(inner_w / crop_w, inner_h / crop_h)
            nw = max(1, int(crop_w * scale))
            nh = max(1, int(crop_h * scale))
            resized = cv2.resize(self.primary_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

            ox = pip_x + 10 + (inner_w - nw) // 2
            oy = pip_y + 26 + (inner_h - nh) // 2
            frame[oy:oy+nh, ox:ox+nw] = resized
            cv2.rectangle(frame, (ox, oy), (ox+nw, oy+nh), COLOR_EMERALD, 1)
            cv2.putText(frame, f"{crop_w}x{crop_h} px", (pip_x + 10, pip_y + pip_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)
        else:
            cv2.putText(frame, "NO FACE CROP", (pip_x + 36, pip_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 90, 105), 1)

        # 6. Cumulative Session Emotion Analytics Panel (Bottom-Right)
        panel_w, panel_h = 190, 160
        panel_x = w - panel_w - 18
        panel_y = pip_y + pip_h + 12

        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_BG_DARK, -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_BORDER, 1)
        cv2.putText(frame, "SESSION DISTRIBUTION", (panel_x + 10, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)

        total_samples = len(self.session_samples)
        for d_idx, em_name in enumerate(DISPLAY_CLASSES):
            count = self.emotion_counter[em_name]
            pct = (count / total_samples) if total_samples > 0 else 0.0
            dy = panel_y + 28 + (d_idx * 17)
            cv2.putText(frame, em_name[:3], (panel_x + 10, dy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_MUTED, 1)
            cv2.rectangle(frame, (panel_x + 40, dy + 3), (panel_x + 140, dy + 10), (30, 36, 46), -1)
            f_w = int(100 * pct)
            if f_w > 0:
                cv2.rectangle(frame, (panel_x + 40, dy + 3), (panel_x + 40 + f_w, dy + 10), EMOTION_COLORS.get(em_name, COLOR_TEXT_WHITE), -1)
            cv2.putText(frame, f"{pct*100:3.0f}%", (panel_x + 146, dy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_WHITE, 1)

        # 7. Bottom Legend Bar
        cv2.rectangle(frame, (0, h - 30), (w, h), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, h - 30), (w, h - 30), COLOR_BORDER, 1)
        legend = "[SPACE]: Start/Stop Camera  |  [S]: Save Snapshot  |  [L]: Export JSON Log  |  [Q]: Quit"
        cv2.putText(frame, legend, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

        return frame

    def run_image(self, image_path: str, no_display: bool = False):
        """Processes and evaluates a static image with integrated YOLO + Emotion recognition."""
        print(f"\n[ImageMode] Loading test photo from: {image_path}")
        if not os.path.exists(image_path):
            print(f"❌ ERROR: File does not exist: {image_path}")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ ERROR: Could not decode image from: {image_path}")
            return

        h, w = frame.shape[:2]
        print(f"[ImageMode] Image dimensions: {w}x{h}")

        # Run face detection
        faces, det_ms = self.detector.detect(frame)
        self.det_latency_ms = det_ms
        self.current_faces_data = []

        cls_ms_total = 0.0
        for idx, f in enumerate(faces):
            xyxy = f['xyxy']
            crop, coords = self.detector.extract_face_crop(frame, xyxy, padding=0.12)
            if crop is not None:
                self.primary_crop = crop
                top_em, conf, probs, cls_ms = self.classifier.predict_face(crop)
                cls_ms_total += cls_ms
                self.current_faces_data.append({
                    'xyxy': xyxy,
                    'det_conf': f['conf'],
                    'top_emotion': top_em,
                    'confidence': conf,
                    'probabilities': probs
                })
                print(f"  Face #{idx+1} (Det: {f['conf']*100:.1f}%): Predicted {top_em} ({conf*100:.1f}%) in {cls_ms:.1f}ms")
                if probs:
                    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("  Top 3:", ", ".join([f"{k}: {v*100:.1f}%" for k, v in top3]))
            else:
                self.current_faces_data.append({
                    'xyxy': xyxy,
                    'det_conf': f['conf'],
                    'top_emotion': None,
                    'confidence': 0.0,
                    'probabilities': None
                })
                print(f"  Face #{idx+1}: EMOTION CLASSIFICATION UNAVAILABLE")

        self.cls_latency_ms = cls_ms_total
        self.total_latency_ms = det_ms + cls_ms_total
        self.camera_active = True

        display = self.draw_hud(frame.copy())
        out_path = f"evaluated_emotion_{os.path.basename(image_path)}"
        cv2.imwrite(out_path, display)
        print(f"[ImageMode] Saved evaluated output to: {out_path}")

        if not no_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, min(w, 1200), min(h, 800))
            cv2.imshow(self.window_name, display)
            print("[ImageMode] Press any key while focused on window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run_webcam(self):
        """Runs the real-time webcam recognition loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n✅ Real-Time YOLO Emotion Recognition System Ready.")
        print("   Click '[> START CAMERA]' or press [SPACEBAR] to begin streaming.")
        print("   Press 'S' to save snapshot.")
        print("   Press 'L' to export session analytics log to JSON.")
        print("   Press 'Q' or [ESC] to quit.\n")

        # Automatically start camera on launch
        self.toggle_camera()

        prev_time = time.time()
        frame_idx = 0
        det_interval = 2  # Evaluate detection every 2nd frame for responsive 30+ FPS

        standby_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        while True:
            frame_idx += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
            prev_time = curr_time

            if self.camera_active:
                raw_frame = self.camera.read()
                if raw_frame is None:
                    time.sleep(0.01)
                    continue

                # Run inference on scheduled interval
                if frame_idx % det_interval == 0 or not self.current_faces_data:
                    t_loop_start = time.perf_counter()
                    faces, det_ms = self.detector.detect(raw_frame)
                    self.det_latency_ms = det_ms

                    new_faces_data = []
                    cls_total_ms = 0.0

                    if faces:
                        for idx, f in enumerate(faces):
                            xyxy = f['xyxy']
                            crop, coords = self.detector.extract_face_crop(raw_frame, xyxy, padding=0.12)
                            if idx == 0:
                                self.primary_crop = crop

                            if crop is not None:
                                prev_ema = self.face_ema_tracker.get(idx)
                                top_em, conf, probs, cls_ms = self.classifier.predict_face(crop, prev_ema)
                                cls_total_ms += cls_ms
                                self.face_ema_tracker[idx] = probs

                                # Record into session history (Section 9)
                                self.emotion_counter[top_em] += 1
                                self.session_samples.append({
                                    "time_ms": int((time.time() - self.session_start_time) * 1000),
                                    "dominant_emotion": top_em,
                                    "confidence": round(conf, 3),
                                    "probabilities": {k: round(v, 3) for k, v in probs.items()},
                                    "detection_status": "FACE DETECTED"
                                })

                                new_faces_data.append({
                                    'xyxy': xyxy,
                                    'det_conf': f['conf'],
                                    'top_emotion': top_em,
                                    'confidence': conf,
                                    'probabilities': probs
                                })
                            else:
                                new_faces_data.append({
                                    'xyxy': xyxy,
                                    'det_conf': f['conf'],
                                    'top_emotion': None,
                                    'confidence': 0.0,
                                    'probabilities': None
                                })
                    else:
                        self.primary_crop = None
                        self.face_ema_tracker.clear()

                    self.current_faces_data = new_faces_data
                    self.cls_latency_ms = cls_total_ms
                    self.total_latency_ms = (time.perf_counter() - t_loop_start) * 1000.0

                display_frame = self.draw_hud(raw_frame.copy())
            else:
                display_frame = self.draw_hud(standby_canvas.copy())

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACEBAR
                self.toggle_camera()
            elif key == ord('s') or key == ord('S'):
                self.save_snapshot(display_frame)
            elif key == ord('l') or key == ord('L'):
                self.export_analytics_log()

        # Cleanup on exit
        if self.camera_active:
            self.camera.stop()
        cv2.destroyAllWindows()
        print("\n[Shutdown] Application closed cleanly.")

def main():
    parser = argparse.ArgumentParser(description="SignalWell Integrated YOLO Facial Emotion Recognition Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--image", type=str, default="", help="Path to static image for single evaluation")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO face detection confidence threshold (default: 0.35)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference resolution (default: 640)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without opening GUI window")
    args = parser.parse_args()

    app = RealTimeYOLOEmotionApp(
        camera_idx=args.camera,
        conf_thresh=args.conf,
        imgsz=args.imgsz
    )

    if args.image:
        app.run_image(args.image, no_display=args.no_display)
    else:
        app.run_webcam()

if __name__ == "__main__":
    main()
