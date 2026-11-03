#!/usr/bin/env python3
"""
SignalWell — Complete Real-Time YOLO Facial Emotion Recognition Application
=============================================================================
Integrates dedicated YOLO11n-face detection with the trained facial emotion
classifier checkpoint 'face_model_v6_best.pt' and full session-based evaluation.

Pipeline Architecture:
  Laptop Webcam
       |
  Continuous Frame Capture (DirectShow Threaded)
       |
  YOLO Face Detection (YOLO11n-face)
       |
  Face Bounding Box & 12% Context Padding
       |
  Face Cropping & Clamping
       |
  Image Preprocessing (160x160 RGB, [-1.0, 1.0] Normalization)
       |
  Emotion Model Inference (InceptionResnetV1 v6)
       |
  Seven-Class Prediction (Ekman 7 Categories)
       |
  Prediction Smoothing (Temporal EMA alpha=0.35)
       |
  Live Emotion Display & Telemetry
       |
  Session Prediction Storage (Timestamped JSON & CSV Records)
       |
  Final Comprehensive Session Evaluation Report & Interactive Visual Dashboard

Features & Controls:
  - Session Duration Selector: 1 Minute (60s) or 2 Minutes (120s) [Keys: 1, 2]
  - Session Control: Start / Stop Session [Key: SPACEBAR], Reset Session [Key: R]
  - Auto-Finalization: Automatically concludes session when duration expires
  - Live HUD: Bounding box, floating emotion badge, probability bars, PiP face crop, FPS & latency
  - Error Handling:
      * No face: "Face not detected" (Never recorded as Neutral)
      * Crop error: "Emotion classification unavailable" (Never recorded)
      * Genuine Neutral: "Neutral" (Recorded as valid Neutral prediction)
      * Camera error: Disconnection alert
      * Insufficient data: Clear reliability verdict banner if face detection < 40%
  - Visual Analytics Dashboard: On-screen modal in OpenCV and standalone interactive HTML report
  - Multi-Format Exports: Structured JSON, Tabular CSV, and Self-Contained HTML Dashboard
"""

import os
import sys
import time
import json
import csv
import argparse
import threading
import webbrowser
from datetime import datetime
from collections import Counter, defaultdict, deque

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
from session_evaluator import (
    SessionEvaluator,
    EMOTION_COLORS_BGR,
    EMOTION_COLORS_HEX
)

# Color Theme (BGR)
COLOR_BG_DARK       = (18, 22, 28)
COLOR_PANEL_BG      = (25, 30, 38)
COLOR_BORDER        = (50, 60, 75)
COLOR_TEXT_WHITE    = (240, 242, 245)
COLOR_TEXT_MUTED    = (150, 160, 175)

# Status Colors (BGR)
COLOR_EMERALD       = (154, 227, 95)    # Success / Active
COLOR_AMBER         = (75, 168, 229)    # Searching / Warning
COLOR_ORANGE        = (22, 115, 249)    # No face
COLOR_CRIMSON       = (68, 68, 239)     # Stop / Error
COLOR_CYAN          = (248, 189, 56)    # Info / Ready
COLOR_VIOLET        = (247, 85, 168)    # Surprise / Multi-face

# Default Application Dimensions
FRAME_WIDTH  = 1140
FRAME_HEIGHT = 720

class WebcamCaptureThread:
    """
    Dedicated thread for webcam capture with Windows DirectShow backend
    for non-blocking, stutter-free 30+ FPS streaming.
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

        # Prefer DirectShow on Windows to eliminate buffer latency
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap or not self.cap.isOpened():
            self.has_error = True
            self.error_msg = f"Cannot connect to webcam at index {self.camera_idx}"
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
    Main Real-Time Facial Emotion Recognition & Session Evaluation Application.
    """
    def __init__(self, camera_idx=0, conf_thresh=0.35, imgsz=640, default_duration=60):
        self.camera_idx = camera_idx
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.target_duration = default_duration  # 60 or 120 seconds

        print("\n==================================================================")
        print(" SIGNALWELL -- REAL-TIME YOLOv11 & EMOTION EVALUATION SYSTEM")
        print("==================================================================")

        # 1. Initialize YOLO Face Detector (loaded once)
        self.detector = YOLOFaceDetector(conf_thresh=self.conf_thresh, imgsz=self.imgsz)

        # 2. Initialize Emotion Classifier (face_model_v6_best.pt loaded once)
        self.classifier = FacialEmotionClassifier(ema_alpha=0.35)

        # 3. Initialize Session Evaluator
        self.evaluator = SessionEvaluator(selected_duration_sec=self.target_duration)

        # 4. Threaded Webcam Capture
        self.camera = WebcamCaptureThread(camera_idx=self.camera_idx)
        self.camera_active = False

        # Status & View Modes
        self.view_mode = "CAMERA"  # "CAMERA" or "DASHBOARD"
        self.det_status = "INITIALIZING CAMERA"
        self.last_exported_html = ""

        # Performance Metrics
        self.fps = 0.0
        self.det_latency_ms = 0.0
        self.cls_latency_ms = 0.0
        self.total_latency_ms = 0.0

        # Face Data & Per-Face EMA Tracking
        self.current_faces_data = []
        self.face_ema_tracker = {}
        self.primary_crop = None

        # Interactive Button Coordinates (x1, y1, x2, y2)
        # Top Action Bar
        self.btn_camera   = (15, 10, 150, 42)
        self.btn_dur_1m   = (160, 10, 230, 42)
        self.btn_dur_2m   = (238, 10, 308, 42)
        self.btn_session  = (318, 10, 480, 42)
        self.btn_reset    = (488, 10, 575, 42)
        self.btn_report   = (583, 10, 695, 42)
        self.btn_snap     = (703, 10, 785, 42)

        # Dashboard View Buttons
        self.btn_dash_new   = (40, 650, 240, 695)
        self.btn_dash_html  = (255, 650, 485, 695)
        self.btn_dash_exp   = (500, 650, 710, 695)
        self.btn_dash_back  = (725, 650, 930, 695)

        self.window_name = "SignalWell -- Real-Time Facial Emotion Recognition & Session Evaluation"

    def toggle_camera(self):
        """Toggles the live webcam feed."""
        if not self.camera_active:
            print("[Webcam] Connecting to laptop webcam...")
            self.det_status = "INITIALIZING CAMERA"
            if self.camera.start():
                self.camera_active = True
                self.det_status = "SEARCHING FOR FACE"
                print("[Webcam] Camera active and streaming.")
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
            print("[Webcam] Camera paused.")

    def set_duration(self, duration_sec: int):
        """Sets target session duration (60s or 120s)."""
        if self.evaluator.state == "RUNNING":
            print("[Session] Cannot change duration while session is active. Reset first.")
            return
        self.target_duration = duration_sec
        self.evaluator.selected_duration_sec = duration_sec
        print(f"[Session] Target duration set to {duration_sec}s ({duration_sec//60} min).")

    def toggle_session(self):
        """Starts or stops the evaluation session."""
        if self.evaluator.state != "RUNNING":
            # Ensure camera is running
            if not self.camera_active:
                self.toggle_camera()
            self.evaluator.start_session(duration_sec=self.target_duration)
            self.view_mode = "CAMERA"
        else:
            print("[Session] Manually stopping session...")
            self._conclude_session()

    def reset_session(self):
        """Resets session state to IDLE."""
        self.evaluator.reset_session()
        self.view_mode = "CAMERA"
        print("[Session] Session reset to IDLE.")

    def _conclude_session(self):
        """Finalizes session, calculates metrics, exports files, and switches to dashboard."""
        report = self.evaluator.finalize_session()

        # Multi-Format Exports
        json_path = self.evaluator.export_json()
        csv_path = self.evaluator.export_csv()
        html_path = self.evaluator.export_html_dashboard()
        self.last_exported_html = html_path

        # Automatically open interactive HTML report in user's default browser
        try:
            abs_html = os.path.abspath(html_path)
            webbrowser.open_new_tab(f"file:///{abs_html}")
            print(f"[Dashboard] Opened interactive HTML report in browser: {abs_html}")
        except Exception as e:
            print(f"[Dashboard] Note: Could not auto-open browser: {e}")

        # Switch to on-screen Visual Analytics Dashboard
        self.view_mode = "DASHBOARD"

    def save_snapshot(self, frame=None):
        """Saves current annotated frame to disk."""
        if frame is None:
            return
        os.makedirs("snapshots", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("snapshots", f"SignalWell_Snapshot_{ts}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"[Snapshot] Saved snapshot: {os.path.abspath(out_path)}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handles interactive on-screen button clicks."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.view_mode == "CAMERA":
            # 1. Camera Toggle
            x1, y1, x2, y2 = self.btn_camera
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.toggle_camera()
                return

            # 2. Duration 1 Min
            x1, y1, x2, y2 = self.btn_dur_1m
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.set_duration(60)
                return

            # 3. Duration 2 Min
            x1, y1, x2, y2 = self.btn_dur_2m
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.set_duration(120)
                return

            # 4. Start / Stop Session
            x1, y1, x2, y2 = self.btn_session
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.toggle_session()
                return

            # 5. Reset Session
            x1, y1, x2, y2 = self.btn_reset
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.reset_session()
                return

            # 6. View Report Dashboard
            x1, y1, x2, y2 = self.btn_report
            if x1 <= x <= x2 and y1 <= y <= y2:
                if self.evaluator.completed_report is not None:
                    self.view_mode = "DASHBOARD"
                elif self.evaluator.state == "RUNNING":
                    self._conclude_session()
                else:
                    print("[Report] No session completed yet. Start a session first.")
                return

            # 7. Snapshot
            x1, y1, x2, y2 = self.btn_snap
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.save_snapshot()
                return

        elif self.view_mode == "DASHBOARD":
            # 1. Start New Session
            x1, y1, x2, y2 = self.btn_dash_new
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.evaluator.reset_session()
                self.view_mode = "CAMERA"
                self.toggle_session()
                return

            # 2. Open HTML Report
            x1, y1, x2, y2 = self.btn_dash_html
            if x1 <= x <= x2 and y1 <= y <= y2:
                if self.last_exported_html and os.path.exists(self.last_exported_html):
                    webbrowser.open_new_tab(f"file:///{os.path.abspath(self.last_exported_html)}")
                return

            # 3. Re-Export Files
            x1, y1, x2, y2 = self.btn_dash_exp
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.evaluator.export_json()
                self.evaluator.export_csv()
                return

            # 4. Back to Camera Feed
            x1, y1, x2, y2 = self.btn_dash_back
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.view_mode = "CAMERA"
                return

    def draw_hud(self, frame):
        """Renders comprehensive HUD with bounding boxes, badges, bars, PiP, and session timer."""
        h, w = frame.shape[:2]

        # 1. Update Detection and Classification Status
        if not self.camera_active:
            if self.camera.has_error:
                self.det_status = "CAMERA DISCONNECTED / ERROR"
                status_color = COLOR_CRIMSON
            else:
                self.det_status = "CAMERA PAUSED"
                status_color = COLOR_AMBER
        else:
            num_faces = len(self.current_faces_data)
            if num_faces == 0:
                self.det_status = "Face not detected"
                status_color = COLOR_ORANGE
            elif num_faces == 1:
                self.det_status = "Face detected"
                status_color = COLOR_EMERALD
            else:
                self.det_status = f"{num_faces} Faces detected"
                status_color = COLOR_VIOLET

        # 2. Header Background Bar
        cv2.rectangle(frame, (0, 0), (w, 82), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, 82), (w, 82), COLOR_BORDER, 1)

        # ----------------------------------------------------
        # Button 1: Camera Toggle
        # ----------------------------------------------------
        bx1, by1, bx2, by2 = self.btn_camera
        btn_bg = COLOR_EMERALD if not self.camera_active else COLOR_CRIMSON
        btn_txt = "[> START CAM]" if not self.camera_active else "[STOP CAM]"
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_bg, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_TEXT_WHITE, 1)
        cv2.putText(frame, btn_txt, (bx1 + 10, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 15, 20), 2)

        # ----------------------------------------------------
        # Buttons 2 & 3: Duration Selector (1 Min / 2 Min)
        # ----------------------------------------------------
        # 1 Min
        d1x1, d1y1, d1x2, d1y2 = self.btn_dur_1m
        bg_1m = COLOR_CYAN if self.target_duration == 60 else COLOR_PANEL_BG
        col_1m = (10, 15, 20) if self.target_duration == 60 else COLOR_TEXT_MUTED
        cv2.rectangle(frame, (d1x1, d1y1), (d1x2, d1y2), bg_1m, -1)
        cv2.rectangle(frame, (d1x1, d1y1), (d1x2, d1y2), COLOR_BORDER, 1)
        cv2.putText(frame, "1 MIN", (d1x1 + 10, d1y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_1m, 2 if self.target_duration == 60 else 1)

        # 2 Min
        d2x1, d2y1, d2x2, d2y2 = self.btn_dur_2m
        bg_2m = COLOR_CYAN if self.target_duration == 120 else COLOR_PANEL_BG
        col_2m = (10, 15, 20) if self.target_duration == 120 else COLOR_TEXT_MUTED
        cv2.rectangle(frame, (d2x1, d2y1), (d2x2, d2y2), bg_2m, -1)
        cv2.rectangle(frame, (d2x1, d2y1), (d2x2, d2y2), COLOR_BORDER, 1)
        cv2.putText(frame, "2 MIN", (d2x1 + 10, d2y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_2m, 2 if self.target_duration == 120 else 1)

        # ----------------------------------------------------
        # Button 4: Start / Stop Session
        # ----------------------------------------------------
        sx1, sy1, sx2, sy2 = self.btn_session
        is_running = (self.evaluator.state == "RUNNING")
        ses_bg = COLOR_CRIMSON if is_running else COLOR_EMERALD
        ses_txt = "[STOP SESSION]" if is_running else "[> START SESSION]"
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), ses_bg, -1)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), COLOR_TEXT_WHITE, 1)
        cv2.putText(frame, ses_txt, (sx1 + 8, sy1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 15, 20), 2)

        # ----------------------------------------------------
        # Button 5: Reset Session
        # ----------------------------------------------------
        rx1, ry1, rx2, ry2 = self.btn_reset
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_BORDER, 1)
        cv2.putText(frame, "RESET (R)", (rx1 + 8, ry1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1)

        # ----------------------------------------------------
        # Button 6: Report / Dashboard
        # ----------------------------------------------------
        repx1, repy1, repx2, repy2 = self.btn_report
        cv2.rectangle(frame, (repx1, repy1), (repx2, repy2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (repx1, repy1), (repx2, repy2), COLOR_BORDER, 1)
        cv2.putText(frame, "REPORT (D)", (repx1 + 8, repy1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1)

        # ----------------------------------------------------
        # Button 7: Snapshot
        # ----------------------------------------------------
        snx1, sny1, snx2, sny2 = self.btn_snap
        cv2.rectangle(frame, (snx1, sny1), (snx2, sny2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (snx1, sny1), (snx2, sny2), COLOR_BORDER, 1)
        cv2.putText(frame, "SNAP (S)", (snx1 + 8, sny1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1)

        # Telemetry HUD (Top-right)
        telemetry = f"FPS: {self.fps:4.1f} | Det: {self.det_latency_ms:4.1f}ms | Cls: {self.cls_latency_ms:4.1f}ms | Tot: {self.total_latency_ms:4.1f}ms"
        cv2.putText(frame, telemetry, (w - 380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_WHITE, 1)

        # ----------------------------------------------------
        # Row 2: Status Banner + Session Countdown Progress Bar
        # ----------------------------------------------------
        # Left: Status badge
        cv2.putText(frame, f"* {self.det_status.upper()} *", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2)

        # Right / Center: Session Countdown & Progress
        elapsed = self.evaluator.get_elapsed_seconds()
        total_d = self.evaluator.selected_duration_sec
        remaining = self.evaluator.get_remaining_seconds()
        pct = min(1.0, elapsed / total_d) if total_d > 0 else 0.0

        mins_el, secs_el = int(elapsed // 60), int(elapsed % 60)
        mins_tot, secs_tot = int(total_d // 60), int(total_d % 60)
        valid_count = len(self.evaluator.samples)

        if self.evaluator.state == "RUNNING":
            timer_text = f"SESSION: {mins_el:02d}:{secs_el:02d} / {mins_tot:02d}:{secs_tot:02d} ({pct*100:3.0f}%) | SAMPLES: {valid_count}"
            prog_col = COLOR_EMERALD
        elif self.evaluator.state == "COMPLETED":
            timer_text = f"SESSION COMPLETED ({self.evaluator.actual_duration_sec:.1f}s) | SAMPLES: {valid_count} | Press [D] to View Report"
            prog_col = COLOR_CYAN
        else:
            timer_text = f"SESSION: READY ({self.target_duration//60} MIN) - Press [SPACE] to Start"
            prog_col = COLOR_TEXT_MUTED

        cv2.putText(frame, timer_text, (260, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_WHITE, 1)

        # Progress bar track & fill
        pb_x1, pb_y1, pb_x2, pb_y2 = 780, 56, w - 20, 70
        cv2.rectangle(frame, (pb_x1, pb_y1), (pb_x2, pb_y2), (30, 36, 46), -1)
        fill_w = int((pb_x2 - pb_x1) * pct)
        if fill_w > 0:
            cv2.rectangle(frame, (pb_x1, pb_y1), (pb_x1 + fill_w, pb_y2), prog_col, -1)
        cv2.rectangle(frame, (pb_x1, pb_y1), (pb_x2, pb_y2), COLOR_BORDER, 1)

        # 3. Standby Screen when Camera is Off
        if not self.camera_active:
            cv2.rectangle(frame, (0, 83), (w, h - 30), (12, 15, 20), -1)
            cv2.putText(frame, "WEBCAM IS PAUSED / STOPPED", (w // 2 - 180, h // 2 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 105, 125), 2)
            cv2.putText(frame, "Click [> START CAM] or press [SPACEBAR] to begin streaming", (w // 2 - 250, h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT_MUTED, 1)
            return frame

        # 4. Render Detected Faces, Emotion Badges, and Lateral Probability Bars
        for item in self.current_faces_data:
            x1, y1, x2, y2 = item['xyxy']
            fw, fh = x2 - x1, y2 - y1
            top_em = item['top_emotion']
            top_conf = item['confidence']
            probs = item['probabilities']

            face_color = EMOTION_COLORS_BGR.get(top_em, COLOR_EMERALD) if top_em else COLOR_AMBER

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
            badge_w = max(220, fw)
            badge_h = 44
            badge_x = max(10, min(w - badge_w - 10, x1 + (fw - badge_w) // 2))
            badge_y = max(88, y1 - badge_h - 10)

            cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), COLOR_BG_DARK, -1)
            cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), face_color, 1)

            if top_em is not None:
                emoji = EMOTION_EMOJIS.get(top_em, "")
                badge_title = f"{emoji} {top_em.upper()} ({top_conf*100:.1f}%)"
                badge_sub = f"DET: {item['det_conf']*100:.1f}% * EMA SMOOTHED"
                cv2.putText(frame, badge_title, (badge_x + 10, badge_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, COLOR_TEXT_WHITE, 2)
                cv2.putText(frame, badge_sub, (badge_x + 10, badge_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, face_color, 1)
            else:
                cv2.putText(frame, "EMOTION CLASSIFICATION UNAVAILABLE", (badge_x + 8, badge_y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CRIMSON, 1)

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
                    cv2.putText(frame, em_name[:3], (bar_x + 8, ry + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_MUTED, 1)
                    cv2.rectangle(frame, (bar_x + 36, ry + 2), (bar_x + 115, ry + 9), (30, 36, 46), -1)
                    fill_w = int(79 * prob_val)
                    bar_col = EMOTION_COLORS_BGR.get(em_name, COLOR_TEXT_WHITE)
                    if fill_w > 0:
                        cv2.rectangle(frame, (bar_x + 36, ry + 2), (bar_x + 36 + fill_w, ry + 9), bar_col, -1)
                    cv2.putText(frame, f"{prob_val*100:3.0f}%", (bar_x + 120, ry + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_WHITE, 1)

        # 5. Picture-in-Picture (PiP) Extracted Face Crop Preview (Top-Right)
        pip_w, pip_h = 190, 190
        pip_x = w - pip_w - 18
        pip_y = 92

        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_BG_DARK, -1)
        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_BORDER, 1)
        cv2.putText(frame, "EXTRACTED CROP (160x160)", (pip_x + 10, pip_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1)

        if self.primary_crop is not None and self.primary_crop.size > 0:
            crop_h, crop_w = self.primary_crop.shape[:2]
            inner_w, inner_h = 170, 145
            scale = min(inner_w / crop_w, inner_h / crop_h)
            nw = max(1, int(crop_w * scale))
            nh = max(1, int(crop_h * scale))
            resized = cv2.resize(self.primary_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

            ox = pip_x + 10 + (inner_w - nw) // 2
            oy = pip_y + 26 + (inner_h - nh) // 2
            frame[oy:oy+nh, ox:ox+nw] = resized
            cv2.rectangle(frame, (ox, oy), (ox+nw, oy+nh), COLOR_EMERALD, 1)
            cv2.putText(frame, f"{crop_w}x{crop_h} px", (pip_x + 10, pip_y + pip_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (pip_x + 30, pip_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 90, 105), 1)

        # 6. Session Distribution Panel (Bottom-Right)
        panel_w, panel_h = 190, 175
        panel_x = w - panel_w - 18
        panel_y = pip_y + pip_h + 12

        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_BG_DARK, -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_BORDER, 1)
        cv2.putText(frame, "SESSION DISTRIBUTION", (panel_x + 10, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1)

        total_samples = len(self.evaluator.samples)
        for d_idx, em_name in enumerate(DISPLAY_CLASSES):
            count = self.evaluator.emotion_counter[em_name]
            pct = (count / total_samples) if total_samples > 0 else 0.0
            dy = panel_y + 28 + (d_idx * 19)
            cv2.putText(frame, em_name[:3], (panel_x + 10, dy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_MUTED, 1)
            cv2.rectangle(frame, (panel_x + 40, dy + 3), (panel_x + 140, dy + 10), (30, 36, 46), -1)
            f_w = int(100 * pct)
            if f_w > 0:
                cv2.rectangle(frame, (panel_x + 40, dy + 3), (panel_x + 40 + f_w, dy + 10), EMOTION_COLORS_BGR.get(em_name, COLOR_TEXT_WHITE), -1)
            cv2.putText(frame, f"{pct*100:3.0f}%", (panel_x + 146, dy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_WHITE, 1)

        # 7. Bottom Legend Bar
        cv2.rectangle(frame, (0, h - 30), (w, h), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, h - 30), (w, h - 30), COLOR_BORDER, 1)
        legend = "[SPACE]: Start/Stop Session  |  [1/2]: 1m/2m Duration  |  [R]: Reset  |  [D]: Report Dashboard  |  [S]: Snapshot  |  [Q]: Quit"
        cv2.putText(frame, legend, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1)

        return frame

    def draw_dashboard_view(self):
        """Renders the comprehensive full-screen Visual Analytics Dashboard modal in OpenCV."""
        canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        w, h = FRAME_WIDTH, FRAME_HEIGHT

        report = self.evaluator.finalize_session()
        info = report["sessionInformation"]
        dist = report["expressionDistribution"]
        mf = report["mostFrequentlyDetectedExpression"]
        conf_ana = report["confidenceAnalysis"]
        quality = report["dataQuality"]
        events = report["temporalAnalysis"]["significantEvents"]

        # 1. Dashboard Header
        cv2.rectangle(canvas, (0, 0), (w, 80), COLOR_BG_DARK, -1)
        cv2.line(canvas, (0, 80), (w, 80), COLOR_BORDER, 1)

        title_txt = f"SIGNALWELL -- EVALUATION REPORT [{info['sessionId']}]"
        cv2.putText(canvas, title_txt, (30, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_TEXT_WHITE, 2)
        sub_txt = f"Target Duration: {info['selectedDurationSeconds']}s ({info['selectedDurationSeconds']//60} min) | Actual: {info['actualDurationSeconds']:.1f}s | Captured: {info['totalCapturedFrames']} frames | Analyzed: {info['totalAnalyzedFrames']}"
        cv2.putText(canvas, sub_txt, (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MUTED, 1)

        # 2. Key Metric Cards (Row 1)
        cards = [
            ("DOMINANT EXPRESSION", f"{mf['emotion'].upper()} ({mf['percentage']:.1f}%)", f"{mf['count']} classified frames", EMOTION_COLORS_BGR.get(mf['emotion'], COLOR_EMERALD)),
            ("TOTAL PREDICTIONS", str(info['totalValidPredictions']), f"Failures: {info['totalClassificationFailures']}", COLOR_CYAN),
            ("AVG CONFIDENCE", f"{conf_ana['overallAverageConfidence']*100:.1f}%", "Neural Softmax Certainty", COLOR_EMERALD),
            ("FACE TRACK RATE", f"{quality['faceDetectionSuccessRatePct']:.1f}%", f"Missing frames: {quality['missingOrInvalidPredictions']}", COLOR_EMERALD if quality['isDataSufficient'] else COLOR_ORANGE)
        ]

        card_w = (w - 60 - 45) // 4
        card_y = 95
        card_h = 95

        for i, (label, val, sub, col) in enumerate(cards):
            cx = 30 + i * (card_w + 15)
            cv2.rectangle(canvas, (cx, card_y), (cx + card_w, card_y + card_h), COLOR_PANEL_BG, -1)
            cv2.rectangle(canvas, (cx, card_y), (cx + card_w, card_y + card_h), COLOR_BORDER, 1)
            cv2.putText(canvas, label, (cx + 14, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)
            cv2.putText(canvas, val, (cx + 14, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            cv2.putText(canvas, sub, (cx + 14, card_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1)

        # 3. Data Quality Banner
        banner_y = 205
        banner_bg = (20, 35, 25) if quality['isDataSufficient'] else (20, 25, 45)
        banner_border = COLOR_EMERALD if quality['isDataSufficient'] else COLOR_ORANGE
        cv2.rectangle(canvas, (30, banner_y), (w - 30, banner_y + 36), banner_bg, -1)
        cv2.rectangle(canvas, (30, banner_y), (w - 30, banner_y + 36), banner_border, 1)
        cv2.putText(canvas, f"STATUS: {quality['status'].upper()}", (45, banner_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, banner_border, 2)

        # 4. Main Body: Emotion Distribution Bars (Left) & Temporal Timeline (Right)
        # Left Panel: Distribution Bars
        dist_w = 520
        dist_x = 30
        dist_y = 255
        dist_h = 370

        cv2.rectangle(canvas, (dist_x, dist_y), (dist_x + dist_w, dist_y + dist_h), COLOR_PANEL_BG, -1)
        cv2.rectangle(canvas, (dist_x, dist_y), (dist_x + dist_w, dist_y + dist_h), COLOR_BORDER, 1)
        cv2.putText(canvas, "7-CATEGORY EXPRESSION DISTRIBUTION", (dist_x + 18, dist_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 2)

        for idx, em in enumerate(DISPLAY_CLASSES):
            pct = dist[em]["percentage"]
            cnt = dist[em]["count"]
            avg_c = conf_ana['byExpression'][em]['averageConfidence'] * 100.0
            ey = dist_y + 55 + (idx * 43)

            cv2.putText(canvas, em, (dist_x + 18, ey + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1)
            cv2.putText(canvas, f"{cnt} frames", (dist_x + 95, ey + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1)

            # Bar track & fill
            bx1, by1 = dist_x + 170, ey + 2
            bx2, by2 = dist_x + 420, ey + 16
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (30, 36, 46), -1)
            bar_len = int((bx2 - bx1) * (pct / 100.0))
            if bar_len > 0:
                cv2.rectangle(canvas, (bx1, by1), (bx1 + bar_len, by2), EMOTION_COLORS_BGR[em], -1)

            cv2.putText(canvas, f"{pct:4.1f}%", (dist_x + 430, ey + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_WHITE, 1)

        # Right Panel: Chronological Timeline Transition Events
        time_x = dist_x + dist_w + 20
        time_w = w - time_x - 30
        time_y = 255
        time_h = 370

        cv2.rectangle(canvas, (time_x, time_y), (time_x + time_w, time_y + time_h), COLOR_PANEL_BG, -1)
        cv2.rectangle(canvas, (time_x, time_y), (time_x + time_w, time_y + time_h), COLOR_BORDER, 1)
        cv2.putText(canvas, "EXPRESSION TRANSITIONS (TIMELINE)", (time_x + 18, time_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 2)

        if events:
            max_show = 8
            for t_idx, ev in enumerate(events[:max_show]):
                ty = time_y + 58 + (t_idx * 36)
                cv2.putText(canvas, ev["timeLabel"], (time_x + 18, ty + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1)
                cv2.putText(canvas, f"-> {ev['emotion'].upper()}", (time_x + 85, ty + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, EMOTION_COLORS_BGR.get(ev['emotion'], COLOR_TEXT_WHITE), 2)
                cv2.putText(canvas, f"({ev['confidence']*100:.1f}%)", (time_x + 235, ty + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)
                cv2.line(canvas, (time_x + 18, ty + 24), (time_x + time_w - 18, ty + 24), (35, 42, 54), 1)
        else:
            cv2.putText(canvas, "No transition events recorded.", (time_x + 20, time_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MUTED, 1)

        # 5. Dashboard Action Buttons (Bottom)
        # New Session Button
        nx1, ny1, nx2, ny2 = self.btn_dash_new
        cv2.rectangle(canvas, (nx1, ny1), (nx2, ny2), COLOR_EMERALD, -1)
        cv2.rectangle(canvas, (nx1, ny1), (nx2, ny2), COLOR_TEXT_WHITE, 1)
        cv2.putText(canvas, "[> NEW SESSION]", (nx1 + 18, ny1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10, 15, 20), 2)

        # Open HTML Report Button
        hx1, hy1, hx2, hy2 = self.btn_dash_html
        cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), COLOR_CYAN, -1)
        cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), COLOR_TEXT_WHITE, 1)
        cv2.putText(canvas, "[OPEN HTML DASHBOARD]", (hx1 + 14, hy1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 15, 20), 2)

        # Export Files Button
        ex1, ey1, ex2, ey2 = self.btn_dash_exp
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), COLOR_PANEL_BG, -1)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), COLOR_BORDER, 1)
        cv2.putText(canvas, "[EXPORT JSON & CSV]", (ex1 + 16, ey1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1)

        # Back to Camera Button
        cx1, cy1, cx2, cy2 = self.btn_dash_back
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), COLOR_PANEL_BG, -1)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), COLOR_BORDER, 1)
        cv2.putText(canvas, "[RETURN TO CAMERA]", (cx1 + 16, cy1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_WHITE, 1)

        return canvas

    def run_image(self, image_path: str, no_display: bool = False):
        """Processes and evaluates a static image with integrated YOLO + Emotion recognition."""
        print(f"\n[ImageMode] Loading test photo from: {image_path}")
        if not os.path.exists(image_path):
            print(f"[ImageMode] ERROR: File does not exist: {image_path}")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ImageMode] ERROR: Could not decode image from: {image_path}")
            return

        h, w = frame.shape[:2]
        print(f"[ImageMode] Image dimensions: {w}x{h}")

        faces, det_ms = self.detector.detect(frame)
        self.det_latency_ms = det_ms
        self.current_faces_data = []

        cls_ms_total = 0.0
        for idx, f in enumerate(faces):
            xyxy = f['xyxy']
            crop, coords = self.detector.extract_face_crop(frame, xyxy, padding=0.15, make_square=True)
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
        """Runs the real-time webcam recognition and session evaluation loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n[OK] Real-Time YOLO Emotion Recognition & Evaluation System Ready.")
        print("   - Click '[> START SESSION]' or press [SPACEBAR] to begin.")
        print("   - Press '1' for 1-minute session | '2' for 2-minute session.")
        print("   - Press 'R' to reset session.")
        print("   - Press 'D' to toggle Evaluation Report Dashboard.")
        print("   - Press 'S' to save snapshot.")
        print("   - Press 'Q' or [ESC] to quit.\n")

        # Automatically start camera on launch
        self.toggle_camera()

        prev_time = time.time()
        frame_idx = 0
        det_interval = 2  # Evaluate detection every 2nd frame for 30+ FPS responsiveness

        standby_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        while True:
            frame_idx += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
            prev_time = curr_time

            # Check if active session duration reached -> Auto-Finalize!
            if self.evaluator.is_time_up():
                print(f"\n[Session] Time expired ({self.evaluator.selected_duration_sec}s). Auto-finalizing...")
                self._conclude_session()

            # Render View
            if self.view_mode == "DASHBOARD":
                display_frame = self.draw_dashboard_view()
            else:
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
                                crop, coords = self.detector.extract_face_crop(raw_frame, xyxy, padding=0.15, make_square=True)
                                if idx == 0:
                                    self.primary_crop = crop

                                if crop is not None:
                                    prev_ema = self.face_ema_tracker.get(idx)
                                    top_em, conf, probs, cls_ms = self.classifier.predict_face(crop, prev_ema)
                                    cls_total_ms += cls_ms
                                    self.face_ema_tracker[idx] = probs

                                    # Record into session evaluator (Strict Error Handling Section 9)
                                    if idx == 0 and self.evaluator.state == "RUNNING":
                                        self.evaluator.record_frame(
                                            has_face=True,
                                            raw_emotion=top_em,
                                            raw_conf=conf,
                                            smoothed_emotion=top_em,
                                            probabilities=probs
                                        )

                                    new_faces_data.append({
                                        'xyxy': xyxy,
                                        'det_conf': f['conf'],
                                        'top_emotion': top_em,
                                        'confidence': conf,
                                        'probabilities': probs
                                    })
                                else:
                                    # Crop failure
                                    if idx == 0 and self.evaluator.state == "RUNNING":
                                        self.evaluator.record_frame(has_face=True, cls_error=True)

                                    new_faces_data.append({
                                        'xyxy': xyxy,
                                        'det_conf': f['conf'],
                                        'top_emotion': None,
                                        'confidence': 0.0,
                                        'probabilities': None
                                    })
                        else:
                            # No face detected: Record for data quality but NEVER record as Neutral
                            if self.evaluator.state == "RUNNING":
                                self.evaluator.record_frame(has_face=False)

                            self.primary_crop = None
                            self.face_ema_tracker.clear()

                        self.current_faces_data = new_faces_data
                        self.cls_latency_ms = cls_total_ms
                        self.total_latency_ms = (time.perf_counter() - t_loop_start) * 1000.0

                    display_frame = self.draw_hud(raw_frame.copy())
                else:
                    display_frame = self.draw_hud(standby_canvas.copy())

            cv2.imshow(self.window_name, display_frame)

            # Keyboard handler
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACEBAR: Start / Stop Session
                self.toggle_session()
            elif key == ord('1'):  # 1: Set 1 Min
                self.set_duration(60)
            elif key == ord('2'):  # 2: Set 2 Min
                self.set_duration(120)
            elif key == ord('r') or key == ord('R'):  # R: Reset Session
                self.reset_session()
            elif key == ord('d') or key == ord('D'):  # D: Toggle Dashboard View
                if self.view_mode == "DASHBOARD":
                    self.view_mode = "CAMERA"
                else:
                    if self.evaluator.completed_report is not None:
                        self.view_mode = "DASHBOARD"
                    elif self.evaluator.state == "RUNNING":
                        self._conclude_session()
            elif key == ord('s') or key == ord('S'):  # S: Save Snapshot
                self.save_snapshot(display_frame)
            elif key == ord('l') or key == ord('L'):  # L: Export Files
                self.evaluator.export_json()
                self.evaluator.export_csv()
                self.evaluator.export_html_dashboard()

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
    parser.add_argument("--duration", type=int, default=60, help="Session duration in seconds (60 or 120, default: 60)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without opening GUI window")
    args = parser.parse_args()

    app = RealTimeYOLOEmotionApp(
        camera_idx=args.camera,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        default_duration=args.duration
    )

    if args.image:
        app.run_image(args.image, no_display=args.no_display)
    else:
        app.run_webcam()

if __name__ == "__main__":
    main()
