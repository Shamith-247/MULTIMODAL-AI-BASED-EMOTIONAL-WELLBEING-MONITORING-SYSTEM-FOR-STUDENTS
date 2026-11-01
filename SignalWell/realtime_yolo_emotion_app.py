#!/usr/bin/env python3
"""
SignalWell — Real-Time YOLO Facial Emotion Recognition Application
===================================================================
A high-performance, responsive desktop application for live webcam facial emotion
recognition powered by YOLOv8n-face and calibrated 7-class emotion classification.

Features:
- Fast YOLO-based multi-face detection (YOLOv8n-face)
- 7 Ekman Emotion Categories: Happy, Sad, Disgust, Surprise, Fear, Angry, Neutral
- Asynchronous threaded capture & decoupled inference for smooth 30+ FPS display
- On-screen Start Camera / Stop Camera interactive controls and hotkeys
- Real-time telemetry: Face Detection time, Emotion Classification time, Total Latency, FPS
- Temporal probability smoothing (EMA, alpha=0.30) to eliminate frame-to-frame flicker
- Cumulative Emotion Distribution panel tracking emotion trends over the session
- CLAHE lighting normalization & context-preserving facial cropping
- Interactive mouse click & keyboard controls (SPACE: Start/Stop, C: Calibration, S: Snapshot, Q: Quit)
"""

import os
import sys
import time
import argparse
import threading
from collections import deque, Counter

import cv2
import numpy as np

# Ensure local imports work cleanly
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
    EMOTION_EMOJIS,
    apply_clahe
)

# UI Theme Palette (Dark Medical/Cyber Glass Aesthetic)
COLOR_BG_DARK = (18, 22, 28)        # #1C1612 BGR
COLOR_PANEL_BG = (25, 30, 38)       # #261E19 BGR
COLOR_BORDER = (60, 72, 88)         # #58483C BGR
COLOR_NEON_MOSS = (143, 227, 160)   # Primary tracking green
COLOR_ACCENT_AMBER = (75, 168, 229) # Warning / Scanning amber
COLOR_ACCENT_BLUE = (229, 168, 75)  # Interface blue
COLOR_TEXT_WHITE = (240, 242, 245)
COLOR_TEXT_MUTED = (160, 168, 178)
COLOR_STOP_RED = (75, 80, 225)      # Stop camera red
COLOR_START_GREEN = (80, 185, 95)   # Start camera green

# Default video dimensions
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

class WebcamCaptureThread:
    """
    Dedicated thread for camera frame polling to prevent OS driver stalls from
    blocking the display or inference pipelines.
    """
    def __init__(self, camera_idx=0):
        self.camera_idx = camera_idx
        self.cap = None
        self.frame = None
        self.running = False
        self.is_active = False
        self.lock = threading.Lock()
        self.thread = None

    def probe_and_open(self):
        """Auto-probes camera indices (0, 1, 2) with Windows DirectShow backend."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        
        indices_to_try = [self.camera_idx] + [i for i in [0, 1, 2] if i != self.camera_idx]
        for idx in indices_to_try:
            for b in backends:
                b_name = "DirectShow" if b == cv2.CAP_DSHOW else "Default"
                cap = cv2.VideoCapture(idx, b)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        self.cap = cap
                        self.camera_idx = idx
                        print(f"[WebcamCapture] Acquired camera index {idx} ({test_frame.shape[1]}x{test_frame.shape[0]}, {b_name})")
                        return True
                    cap.release()
        return False

    def start(self):
        if self.probe_and_open():
            self.running = True
            self.is_active = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            return True
        return False

    def _update_loop(self):
        while self.running:
            if self.is_active and self.cap is not None and self.cap.isOpened():
                ret, f = self.cap.read()
                if ret and f is not None:
                    # Mirror flip for intuitive live user interaction
                    f = cv2.flip(f, 1)
                    with self.lock:
                        self.frame = f
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.03)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def toggle_active(self):
        self.is_active = not self.is_active
        return self.is_active

    def stop(self):
        self.running = False
        self.is_active = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class RealTimeYOLOEmotionApp:
    """
    Main Application orchestrator combining YOLO face detection, emotion prediction,
    real-time GUI HUD overlay, and performance telemetry.
    """
    def __init__(self, camera_idx=0, conf_thresh=0.45, imgsz=640):
        self.camera_idx = camera_idx
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Initialize Models
        print("\n" + "="*70)
        print(" SIGNALWELL — REAL-TIME YOLO FACIAL EMOTION RECOGNITION SYSTEM")
        print("="*70)
        self.detector = YOLOFaceDetector(conf_thresh=self.conf_thresh, imgsz=self.imgsz)
        self.classifier = FacialEmotionClassifier(ema_alpha=0.30)

        # Webcam Capture Thread
        self.camera = WebcamCaptureThread(camera_idx=self.camera_idx)

        # Performance Telemetry & Rolling Counters
        self.fps = 0.0
        self.det_latency = 0.0
        self.cls_latency = 0.0
        self.total_latency = 0.0
        self.frame_count = 0

        # Emotion session history for distribution panel
        self.session_emotion_history = deque(maxlen=300)  # rolling history of detected expressions
        self.face_ema_tracker = {}  # tracked EMAs per detected face index

        # Interaction & Control States
        self.camera_active = True
        self.calibration_active = True
        self.window_name = "SignalWell — Real-Time YOLO Facial Emotion Recognition"

        # On-screen button boundaries for mouse clicks: (x1, y1, x2, y2)
        self.btn_camera = (20, 52, 180, 88)
        self.btn_calib = (195, 52, 335, 88)
        self.btn_snap = (350, 52, 475, 88)

    def mouse_callback(self, event, x, y, flags, param):
        """Handles interactive mouse clicks on UI control buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click: Start / Stop Camera Button
            bx1, by1, bx2, by2 = self.btn_camera
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self.toggle_camera()

            # Click: Toggle Calibration
            cx1, cy1, cx2, cy2 = self.btn_calib
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                self.toggle_calibration()

            # Click: Snapshot
            sx1, sy1, sx2, sy2 = self.btn_snap
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                self.save_snapshot()

    def toggle_camera(self):
        self.camera_active = self.camera.toggle_active()
        status_str = "ACTIVE (Running)" if self.camera_active else "STOPPED (Standby)"
        print(f"[Controls] Camera toggled: {status_str}")

    def toggle_calibration(self):
        self.calibration_active = not self.calibration_active
        self.classifier.calibrated = self.calibration_active
        print(f"[Controls] Logit calibration toggled: {self.calibration_active}")

    def save_snapshot(self, frame=None):
        if frame is None:
            frame = self.camera.read()
        if frame is not None:
            os.makedirs("snapshots", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join("snapshots", f"SignalWell_Face_{ts}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[Snapshot] Saved frame snapshot to: {snap_path}")

    def draw_hud(self, frame, faces_data, fps, det_ms, cls_ms, total_ms):
        """
        Renders a high-tech on-screen HUD:
        - Top telemetry bar (FPS, Latencies, Face Count)
        - Interactive Control Buttons (Start/Stop, Calibration, Snapshot)
        - Corner brackets & bounding boxes
        - Floating Emotion Badge with emoji, emotion name, and confidence %
        - Lateral probability breakdown bars for all 7 emotions
        - Rolling Emotion Distribution Panel (historical frequency)
        - Static centering guide box
        """
        h, w = frame.shape[:2]

        # 1. Top Telemetry Bar (Dark Frosted Glass)
        cv2.rectangle(frame, (0, 0), (w, 44), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, 44), (w, 44), COLOR_BORDER, 1)

        sys_title = "SignalWell AI | YOLOv8-Face Real-Time Emotion Monitor"
        cv2.putText(frame, sys_title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_TEXT_WHITE, 2)

        telemetry_str = f"FPS: {fps:4.1f} | Det: {det_ms:4.1f}ms | Cls: {cls_ms:4.1f}ms | Latency: {total_ms:4.1f}ms"
        t_size = cv2.getTextSize(telemetry_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.putText(frame, telemetry_str, (w - t_size[0] - 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_NEON_MOSS, 1)

        # 2. Interactive Control Buttons
        # Camera Start/Stop Button
        cam_color = COLOR_START_GREEN if not self.camera_active else COLOR_STOP_RED
        cam_label = "▶ START CAMERA" if not self.camera_active else "■ STOP CAMERA"
        cv2.rectangle(frame, (self.btn_camera[0], self.btn_camera[1]), (self.btn_camera[2], self.btn_camera[3]), cam_color, -1)
        cv2.rectangle(frame, (self.btn_camera[0], self.btn_camera[1]), (self.btn_camera[2], self.btn_camera[3]), COLOR_TEXT_WHITE, 1)
        cv2.putText(frame, cam_label, (self.btn_camera[0] + 12, self.btn_camera[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        # Calibration Toggle Button
        calib_color = COLOR_PANEL_BG
        calib_label = "CALIB: ON" if self.calibration_active else "CALIB: OFF"
        cv2.rectangle(frame, (self.btn_calib[0], self.btn_calib[1]), (self.btn_calib[2], self.btn_calib[3]), calib_color, -1)
        cv2.rectangle(frame, (self.btn_calib[0], self.btn_calib[1]), (self.btn_calib[2], self.btn_calib[3]), COLOR_BORDER, 1)
        cv2.putText(frame, calib_label, (self.btn_calib[0] + 16, self.btn_calib[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1)

        # Snapshot Button
        cv2.rectangle(frame, (self.btn_snap[0], self.btn_snap[1]), (self.btn_snap[2], self.btn_snap[3]), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (self.btn_snap[0], self.btn_snap[1]), (self.btn_snap[2], self.btn_snap[3]), COLOR_BORDER, 1)
        cv2.putText(frame, "💾 SNAPSHOT", (self.btn_snap[0] + 12, self.btn_snap[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT_WHITE, 1)

        # Standby Overlay if Camera is Stopped
        if not self.camera_active:
            cv2.rectangle(frame, (0, 95), (w, h), (10, 12, 16), -1)
            msg1 = "CAMERA FEED PAUSED (STANDBY)"
            msg2 = "Click 'START CAMERA' or press [SPACEBAR] to resume"
            sz1 = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
            sz2 = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(frame, msg1, ((w - sz1[0]) // 2, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_TEXT_WHITE, 2)
            cv2.putText(frame, msg2, ((w - sz2[0]) // 2, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ACCENT_AMBER, 1)
            return frame

        # 3. Static Center Guide Box (when no face or guide active)
        gw, gh = int(w * 0.45), int(h * 0.62)
        gx1, gy1 = (w - gw) // 2, (h - gh) // 2 + 20
        gx2, gy2 = gx1 + gw, gy1 + gh

        face_count = len(faces_data)
        if face_count == 0:
            # Draw dashed-style static guide box in amber
            dash_len = 16
            for x in range(gx1, gx2, dash_len * 2):
                cv2.line(frame, (x, gy1), (min(x + dash_len, gx2), gy1), COLOR_ACCENT_AMBER, 2)
                cv2.line(frame, (x, gy2), (min(x + dash_len, gx2), gy2), COLOR_ACCENT_AMBER, 2)
            for y in range(gy1, gy2, dash_len * 2):
                cv2.line(frame, (gx1, y), (gx1, min(y + dash_len, gy2)), COLOR_ACCENT_AMBER, 2)
                cv2.line(frame, (gx2, y), (gx2, min(y + dash_len, gy2)), COLOR_ACCENT_AMBER, 2)

            banner = "SCANNING: Center your face in camera view"
            bsz = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
            bx = (w - bsz[0]) // 2
            by = gy1 + gh // 2
            cv2.rectangle(frame, (bx - 12, by - 24), (bx + bsz[0] + 12, by + 10), COLOR_PANEL_BG, -1)
            cv2.rectangle(frame, (bx - 12, by - 24), (bx + bsz[0] + 12, by + 10), COLOR_ACCENT_AMBER, 1)
            cv2.putText(frame, banner, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_TEXT_WHITE, 2)

        # 4. Render Detected Faces
        for item in faces_data:
            x1, y1, x2, y2 = item['xyxy']
            fw, fh = x2 - x1, y2 - y1
            top_em = item['top_emotion']
            top_conf = item['confidence']
            probs = item['probabilities']

            # Corner Reticle Brackets
            corner_len = int(min(fw, fh) * 0.22)
            thick = 3
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), COLOR_NEON_MOSS, thick)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), COLOR_NEON_MOSS, thick)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), COLOR_NEON_MOSS, thick)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), COLOR_NEON_MOSS, thick)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), COLOR_NEON_MOSS, thick)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), COLOR_NEON_MOSS, thick)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), COLOR_NEON_MOSS, thick)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), COLOR_NEON_MOSS, thick)

            # Floating Emotion Badge
            emoji = EMOTION_EMOJIS.get(top_em, '')
            badge_text = f"{emoji} {top_em.upper()} ({top_conf*100:.1f}%)"
            badge_sz = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            badge_x = x1
            badge_y = max(55, y1 - 12)
            cv2.rectangle(frame, (badge_x, badge_y - badge_sz[1] - 8), (badge_x + badge_sz[0] + 16, badge_y + 4), COLOR_PANEL_BG, -1)
            cv2.rectangle(frame, (badge_x, badge_y - badge_sz[1] - 8), (badge_x + badge_sz[0] + 16, badge_y + 4), COLOR_NEON_MOSS, 1)
            cv2.putText(frame, badge_text, (badge_x + 8, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_WHITE, 2)

            # Lateral Live Emotion Probability Bars (Right-side HUD)
            hud_x = min(w - 230, x2 + 16) if x2 + 240 < w else max(10, x1 - 230)
            hud_y = max(105, y1)
            cv2.rectangle(frame, (hud_x - 10, hud_y - 10), (hud_x + 215, hud_y + 205), COLOR_PANEL_BG, -1)
            cv2.rectangle(frame, (hud_x - 10, hud_y - 10), (hud_x + 215, hud_y + 205), COLOR_BORDER, 1)

            cv2.putText(frame, "PREDICTED EMOTIONS", (hud_x, hud_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_WHITE, 1)

            # Sort probabilities descending
            sorted_probs = sorted(probs.items(), key=lambda kv: -kv[1])
            for i, (em_name, p) in enumerate(sorted_probs):
                row_y = hud_y + 32 + i * 24
                bar_w = int(p * 90)
                is_top = (em_name == top_em)
                bar_col = COLOR_NEON_MOSS if is_top else COLOR_BORDER

                cv2.putText(frame, f"{em_name[:7]:<7}", (hud_x, row_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1)
                cv2.rectangle(frame, (hud_x + 65, row_y), (hud_x + 65 + bar_w, row_y + 12), bar_col, -1)
                cv2.putText(frame, f"{int(p*100)}%", (hud_x + 165, row_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_WHITE, 1)

        # 5. Session Emotion Distribution Panel (Bottom-Left)
        if len(self.session_emotion_history) > 0:
            p_x = 20
            p_y = h - 130
            pw, ph = 270, 115
            cv2.rectangle(frame, (p_x, p_y), (p_x + pw, p_y + ph), COLOR_PANEL_BG, -1)
            cv2.rectangle(frame, (p_x, p_y), (p_x + pw, p_y + ph), COLOR_BORDER, 1)

            cv2.putText(frame, f"SESSION DISTRIBUTION (n={len(self.session_emotion_history)})", (p_x + 10, p_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_WHITE, 1)

            counts = Counter(self.session_emotion_history)
            total_n = len(self.session_emotion_history)
            top_3 = counts.most_common(3)
            for idx, (em_name, count) in enumerate(top_3):
                pct = count / total_n * 100
                em_y = p_y + 40 + idx * 24
                bar_w = int((pct / 100.0) * 120)
                cv2.putText(frame, f"{em_name[:7]:<7}", (p_x + 10, em_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)
                cv2.rectangle(frame, (p_x + 75, em_y), (p_x + 75 + bar_w, em_y + 12), COLOR_ACCENT_BLUE, -1)
                cv2.putText(frame, f"{pct:.0f}%", (p_x + 205, em_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1)

        # 6. Bottom Key Legend
        legend_str = "[SPACE]: Start/Stop Camera  |  [C]: Toggle Calibration  |  [S]: Snapshot  |  [Q]: Quit"
        cv2.putText(frame, legend_str, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

        return frame

    def run_image(self, image_path, no_display: bool = False):
        """Processes and evaluates a static image with YOLO detection & emotion prediction."""
        print(f"\n[ImageMode] Loading image from {image_path}...")
        if not os.path.exists(image_path):
            print(f"❌ ERROR: File does not exist: {image_path}")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ ERROR: Could not decode image from {image_path}")
            return

        h, w = frame.shape[:2]
        print(f"[ImageMode] Image dimensions: {w}x{h}")

        # Run YOLO face detection
        faces, det_ms = self.detector.detect(frame)
        print(f"[ImageMode] Detected {len(faces)} face(s) in {det_ms:.2f} ms")

        faces_data = []
        cls_ms_total = 0.0

        for idx, f in enumerate(faces):
            x1, y1, x2, y2 = f['xyxy']
            # Apply 15% context-preserving padding
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            top_em, conf, probs, cls_ms = self.classifier.predict_face(crop)
            cls_ms_total += cls_ms

            faces_data.append({
                'xyxy': (x1, y1, x2, y2),
                'top_emotion': top_em,
                'confidence': conf,
                'probabilities': probs
            })

            print(f"  Face #{idx+1}: {top_em} ({conf*100:.1f}%) | Cls latency: {cls_ms:.2f} ms")
            for k in DISPLAY_CLASSES:
                print(f"    - {k:<9}: {probs[k]*100:.1f}%")

        # Render HUD on image
        display = self.draw_hud(frame, faces_data, fps=0.0, det_ms=det_ms, cls_ms=cls_ms_total, total_ms=det_ms + cls_ms_total)
        out_path = f"evaluated_{os.path.basename(image_path)}"
        cv2.imwrite(out_path, display)
        print(f"[ImageMode] Saved annotated output to {out_path}")

        if not no_display:
            cv2.imshow("SignalWell Static Image Evaluation (Press any key to close)", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run_webcam(self):
        """Runs the real-time webcam recognition loop with asynchronous inference."""
        if not self.camera.start():
            print("\n❌ CRITICAL: Could not open built-in laptop webcam.")
            print("   Troubleshooting checklist:")
            print("   1. Verify no other app (Teams, Zoom, Camera app) is using the camera.")
            print("   2. Windows Settings > Privacy & security > Camera > Enable 'Let desktop apps access your camera'.")
            print("   3. Try running with --camera 1 or --camera 2 if external webcam is attached.")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n✅ Webcam opened successfully.")
        print("   Press [SPACEBAR] to Start/Stop camera stream.")
        print("   Press 'c' to toggle logit calibration.")
        print("   Press 's' to save a snapshot.")
        print("   Press 'q' or [ESC] to quit.\n")

        prev_time = time.time()
        det_interval = 2  # Run YOLO detection every 2nd frame for 30+ FPS responsiveness
        cached_faces_data = []

        while True:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            curr_time = time.time()
            self.fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            h, w = frame.shape[:2]

            # Run detection & emotion prediction on controlled interval
            if self.camera_active and (self.frame_count % det_interval == 0 or not cached_faces_data):
                faces, det_ms = self.detector.detect(frame)
                self.det_latency = det_ms

                cls_start = time.perf_counter()
                current_faces_data = []

                for idx, f in enumerate(faces):
                    x1, y1, x2, y2 = f['xyxy']
                    # Context padding
                    pad_x = int((x2 - x1) * 0.15)
                    pad_y = int((y2 - y1) * 0.15)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w, x2 + pad_x)
                    cy2 = min(h, y2 + pad_y)

                    face_crop = frame[cy1:cy2, cx1:cx2]
                    prev_ema = self.face_ema_tracker.get(idx)
                    top_em, conf, probs, _ = self.classifier.predict_face(face_crop, prev_ema)
                    self.face_ema_tracker[idx] = probs
                    self.session_emotion_history.append(top_em)

                    current_faces_data.append({
                        'xyxy': (x1, y1, x2, y2),
                        'top_emotion': top_em,
                        'confidence': conf,
                        'probabilities': probs
                    })

                self.cls_latency = (time.perf_counter() - cls_start) * 1000.0
                self.total_latency = self.det_latency + self.cls_latency
                cached_faces_data = current_faces_data

            # Draw HUD
            display_frame = self.draw_hud(
                frame,
                cached_faces_data,
                self.fps,
                self.det_latency,
                self.cls_latency,
                self.total_latency
            )

            cv2.imshow(self.window_name, display_frame)

            # Keyboard handler
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACEBAR
                self.toggle_camera()
            elif key == ord('c') or key == ord('C'):
                self.toggle_calibration()
            elif key == ord('s') or key == ord('S'):
                self.save_snapshot(display_frame)

        # Cleanup
        self.camera.stop()
        cv2.destroyAllWindows()
        print("\n[Shutdown] Camera released and display closed cleanly.")

def main():
    parser = argparse.ArgumentParser(description="SignalWell Real-Time YOLO Facial Emotion Recognition Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--image", type=str, default="", help="Path to static image for single evaluation")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO face detection confidence threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference resolution (default: 640)")
    parser.add_argument("--no-display", action="store_true", help="Do not open GUI window (useful for headless / batch evaluation)")
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
