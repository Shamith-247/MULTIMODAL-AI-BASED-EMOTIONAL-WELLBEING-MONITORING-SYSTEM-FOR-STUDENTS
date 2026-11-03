#!/usr/bin/env python3
"""
SignalWell — Dedicated YOLO Face Detection Diagnostic Application
==================================================================
Standalone face-detection testing interface featuring YOLO11n-face.

Features:
- Live built-in laptop webcam stream with threaded DirectShow acquisition.
- Interactive Start Camera / Stop Camera button (mouse click & SPACEBAR).
- Accurate face bounding-box rendering with corner accents.
- Separate Picture-in-Picture (PiP) preview of the extracted face crop for debugging.
- Real-time telemetry: FPS, Detection Latency (ms), Number of faces, Confidence (%).
- Explicit status banner with required states:
    * INITIALIZING CAMERA
    * SEARCHING FOR FACE
    * FACE DETECTED
    * MULTIPLE FACES DETECTED
    * NO FACE DETECTED
    * CAMERA ERROR
- Configurable confidence threshold (starts at 0.35, adjustable with +/-).
- Zero emotion classification logic (phase-isolated for face detection verification).
- Static image diagnostic mode (--image path [--no-display]).
"""

import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SignalWell.yolo_face_detector import YOLOFaceDetector

# Color Palette (BGR)
COLOR_BG_DARK       = (16, 20, 26)       # #1A1410
COLOR_PANEL_BG      = (24, 30, 40)       # #281E18
COLOR_BORDER        = (45, 55, 70)       # #46372D
COLOR_TEXT_WHITE    = (245, 245, 245)
COLOR_TEXT_MUTED    = (165, 175, 190)

# Status Colors (BGR)
COLOR_CYAN          = (248, 189, 56)     # INITIALIZING CAMERA
COLOR_AMBER         = (75, 168, 229)     # SEARCHING FOR FACE
COLOR_EMERALD       = (154, 227, 95)     # FACE DETECTED (#5FE39A)
COLOR_VIOLET        = (247, 85, 168)     # MULTIPLE FACES DETECTED
COLOR_ORANGE        = (22, 115, 249)     # NO FACE DETECTED
COLOR_CRIMSON       = (68, 68, 239)      # CAMERA ERROR

# UI Dimensions
FRAME_WIDTH  = 1120
FRAME_HEIGHT = 680

class WebcamCaptureThread:
    """
    Asynchronous webcam capture thread with DirectShow backend.
    Ensures 30+ FPS preview without blocking the UI.
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

        # Try DirectShow on Windows, then standard VideoCapture
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap or not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.camera_idx)

        if not self.cap or not self.cap.isOpened():
            self.has_error = True
            self.error_msg = f"Cannot open webcam (Index {self.camera_idx})"
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

class StandaloneFaceDetectorApp:
    """
    Dedicated Face Detection Diagnostic Application.
    """
    def __init__(self, camera_idx=0, conf_thresh=0.35, imgsz=640):
        self.camera_idx = camera_idx
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Initialize detector
        print("\n=======================================================")
        print(" SIGNALWELL — STANDALONE YOLO FACE DETECTION MONITOR")
        print("=======================================================")
        self.detector = YOLOFaceDetector(conf_thresh=self.conf_thresh, imgsz=self.imgsz)

        # Threaded Camera
        self.camera = WebcamCaptureThread(camera_idx=self.camera_idx)
        self.camera_active = False

        # Status tracking
        self.status = "INITIALIZING CAMERA"
        self.status_color = COLOR_CYAN

        # Performance Metrics
        self.fps = 0.0
        self.det_latency_ms = 0.0
        self.detected_faces = []
        self.primary_crop = None
        self.crop_coords = None

        # Interactive button bounding boxes
        self.btn_camera_rect = (20, 14, 175, 48)   # (x1, y1, x2, y2)
        self.btn_minus_rect  = (200, 14, 235, 48)
        self.btn_plus_rect   = (340, 14, 375, 48)
        self.btn_snap_rect   = (395, 14, 520, 48)

        self.window_name = "SignalWell — Standalone YOLO Face Detector"

    def toggle_camera(self):
        """Starts or stops the live webcam feed."""
        if not self.camera_active:
            print("[Webcam] Initializing built-in laptop camera...")
            self.status = "INITIALIZING CAMERA"
            self.status_color = COLOR_CYAN
            success = self.camera.start()
            if success:
                self.camera_active = True
                self.status = "SEARCHING FOR FACE"
                self.status_color = COLOR_AMBER
                print("[Webcam] Camera started successfully.")
            else:
                self.camera_active = False
                self.status = "CAMERA ERROR"
                self.status_color = COLOR_CRIMSON
                print(f"[Webcam] Error: {self.camera.error_msg}")
        else:
            self.camera_active = False
            self.camera.stop()
            self.status = "NO FACE DETECTED"
            self.status_color = COLOR_ORANGE
            self.detected_faces = []
            self.primary_crop = None
            print("[Webcam] Camera stopped.")

    def adjust_confidence(self, delta):
        """Increases or decreases confidence threshold by delta."""
        self.conf_thresh = max(0.10, min(0.90, round(self.conf_thresh + delta, 2)))
        self.detector.set_confidence_threshold(self.conf_thresh)
        print(f"[Detector] Confidence threshold set to: {self.conf_thresh:.2f}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handles interactive on-screen button clicks."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Start / Stop Camera button
        x1, y1, x2, y2 = self.btn_camera_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.toggle_camera()
            return

        # Confidence [-]
        x1, y1, x2, y2 = self.btn_minus_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.adjust_confidence(-0.05)
            return

        # Confidence [+]
        x1, y1, x2, y2 = self.btn_plus_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.adjust_confidence(+0.05)
            return

        # Save Snapshot
        x1, y1, x2, y2 = self.btn_snap_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.save_snapshot()
            return

    def save_snapshot(self, frame=None):
        """Saves current annotated snapshot to disk."""
        if frame is None:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"face_detection_snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved snapshot: {os.path.abspath(filename)}")

    def render_overlay(self, frame):
        """Renders bounding boxes, HUD overlay, status banner, and face-crop PiP."""
        h, w = frame.shape[:2]

        # 1. Update Detection Status
        if not self.camera_active:
            if self.camera.has_error:
                self.status = "CAMERA ERROR"
                self.status_color = COLOR_CRIMSON
            else:
                self.status = "SEARCHING FOR FACE"
                self.status_color = COLOR_AMBER
        else:
            num_faces = len(self.detected_faces)
            if num_faces == 0:
                self.status = "NO FACE DETECTED"
                self.status_color = COLOR_ORANGE
            elif num_faces == 1:
                self.status = "FACE DETECTED"
                self.status_color = COLOR_EMERALD
            else:
                self.status = "MULTIPLE FACES DETECTED"
                self.status_color = COLOR_VIOLET

        # 2. Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 64), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, 64), (w, 64), COLOR_BORDER, 1)

        # 3. Interactive Start/Stop Camera Button
        bx1, by1, bx2, by2 = self.btn_camera_rect
        btn_bg = COLOR_EMERALD if not self.camera_active else COLOR_CRIMSON
        btn_text = "[> START CAMERA]" if not self.camera_active else "[STOP CAMERA]"
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_bg, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_TEXT_WHITE, 1)
        cv2.putText(frame, btn_text, (bx1 + 10, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10, 15, 20), 2)

        # 4. Confidence Threshold Stepper [-] 0.35 [+]
        mx1, my1, mx2, my2 = self.btn_minus_rect
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), COLOR_BORDER, 1)
        cv2.putText(frame, "-", (mx1 + 12, my1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT_WHITE, 2)

        cv2.putText(frame, f"Conf: {self.conf_thresh:.2f}", (mx2 + 10, my1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT_WHITE, 1)

        px1, py1, px2, py2 = self.btn_plus_rect
        cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_BORDER, 1)
        cv2.putText(frame, "+", (px1 + 10, py1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_TEXT_WHITE, 2)

        # 5. Snapshot Button
        sx1, sy1, sx2, sy2 = self.btn_snap_rect
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), COLOR_PANEL_BG, -1)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), COLOR_BORDER, 1)
        cv2.putText(frame, "SNAPSHOT (S)", (sx1 + 10, sy1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_MUTED, 1)

        # 6. Central Status Banner
        status_text = f"* {self.status} *"
        stat_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        stat_x = max(540, (w - stat_size[0]) // 2)
        stat_y = 38
        cv2.rectangle(frame, (stat_x - 14, 12), (stat_x + stat_size[0] + 14, 52), (12, 16, 22), -1)
        cv2.rectangle(frame, (stat_x - 14, 12), (stat_x + stat_size[0] + 14, 52), self.status_color, 2)
        cv2.putText(frame, status_text, (stat_x, stat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.status_color, 2)

        # 7. Telemetry HUD (Top-right)
        telemetry_str = f"FPS: {self.fps:4.1f} | Det: {self.det_latency_ms:4.1f}ms | Faces: {len(self.detected_faces)}"
        cv2.putText(frame, telemetry_str, (w - 360, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT_WHITE, 1)

        # 8. Draw Bounding Boxes on Detected Faces
        for idx, f in enumerate(self.detected_faces):
            x1, y1, x2, y2 = f['xyxy']
            conf = f['conf']
            fw = x2 - x1
            fh = y2 - y1

            # Box color changes to Violet if multiple faces
            box_color = COLOR_EMERALD if len(self.detected_faces) == 1 else COLOR_VIOLET

            # Bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # High-tech corner accents
            corner_len = max(12, int(min(fw, fh) * 0.18))
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

            # Face label tag: FACE #1: 89.4%
            label = f"FACE #{idx+1}: {conf*100:.1f}%"
            lbl_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
            lbl_y1 = max(68, y1 - 24)
            cv2.rectangle(frame, (x1, lbl_y1), (x1 + lbl_size[0] + 12, lbl_y1 + 22), COLOR_BG_DARK, -1)
            cv2.rectangle(frame, (x1, lbl_y1), (x1 + lbl_size[0] + 12, lbl_y1 + 22), box_color, 1)
            cv2.putText(frame, label, (x1 + 6, lbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT_WHITE, 1)

        # 9. Picture-in-Picture (PiP) Extracted Face Crop Preview (Top-Right)
        pip_w, pip_h = 190, 200
        pip_x = w - pip_w - 18
        pip_y = 74

        # Background panel for face crop
        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_BG_DARK, -1)
        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_BORDER, 1)
        cv2.putText(frame, "EXTRACTED CROP (DEBUG)", (pip_x + 10, pip_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)

        if self.primary_crop is not None and self.primary_crop.size > 0:
            crop_h, crop_w = self.primary_crop.shape[:2]
            # Maintain aspect ratio within inner 170x150 region
            inner_w, inner_h = 170, 150
            scale = min(inner_w / crop_w, inner_h / crop_h)
            nw = max(1, int(crop_w * scale))
            nh = max(1, int(crop_h * scale))
            resized_crop = cv2.resize(self.primary_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

            ox = pip_x + 10 + (inner_w - nw) // 2
            oy = pip_y + 26 + (inner_h - nh) // 2
            frame[oy:oy+nh, ox:ox+nw] = resized_crop
            cv2.rectangle(frame, (ox, oy), (ox+nw, oy+nh), COLOR_EMERALD, 1)

            # Crop resolution info
            crop_info = f"{crop_w}x{crop_h} px"
            cv2.putText(frame, crop_info, (pip_x + 10, pip_y + pip_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)
        else:
            # Placeholder text
            cv2.putText(frame, "NO FACE CROP", (pip_x + 36, pip_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 90, 105), 1)

        # 10. Bottom Key Legend Bar
        cv2.rectangle(frame, (0, h - 30), (w, h), COLOR_BG_DARK, -1)
        cv2.line(frame, (0, h - 30), (w, h - 30), COLOR_BORDER, 1)
        legend_str = "[SPACE]: Start/Stop Camera  |  [+ / -]: Adjust Conf (0.35)  |  [S]: Save Snapshot  |  [Q]: Quit"
        cv2.putText(frame, legend_str, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

        return frame

    def run_image(self, image_path: str, no_display: bool = False):
        """Processes and evaluates a static image with YOLO face detector."""
        print(f"\n[ImageMode] Loading test image from: {image_path}")
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
        self.detected_faces = faces

        print(f"[ImageMode] Detected {len(faces)} face(s) in {det_ms:.2f} ms:")
        for idx, f in enumerate(faces):
            x1, y1, x2, y2 = f['xyxy']
            conf = f['conf'] * 100
            print(f"  Face #{idx+1}: Box=[{x1}, {y1}, {x2}, {y2}] | Confidence: {conf:.1f}%")

        # Extract primary crop
        if faces:
            crop, coords = self.detector.extract_face_crop(frame, faces[0]['xyxy'], padding=0.12)
            self.primary_crop = crop
            self.crop_coords = coords

        self.camera_active = True
        display = self.render_overlay(frame.copy())
        out_path = f"evaluated_face_{os.path.basename(image_path)}"
        cv2.imwrite(out_path, display)
        print(f"[ImageMode] Successfully saved annotated diagnostic output to: {out_path}")

        if not no_display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, min(w, 1200), min(h, 800))
            cv2.imshow(self.window_name, display)
            print("[ImageMode] Press any key while focused on window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run_webcam(self):
        """Runs the continuous webcam loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, FRAME_WIDTH, FRAME_HEIGHT)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n✅ Standalone Face Detector Initialized.")
        print("   Click '[> START CAMERA]' on-screen or press [SPACEBAR] to start webcam.")
        print("   Press '+' or '-' to adjust detection confidence.")
        print("   Press 'S' to save a snapshot.")
        print("   Press 'Q' or [ESC] to quit.\n")

        # Auto-start camera on launch as requested
        self.toggle_camera()

        prev_time = time.time()
        frame_idx = 0
        det_interval = 2  # Evaluate detection every 2nd frame for 30+ FPS responsiveness

        # Blank standby canvas when camera is off
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

                # Run YOLO face detection on interval or if no faces cached
                if frame_idx % det_interval == 0 or not self.detected_faces:
                    faces, det_ms = self.detector.detect(raw_frame)
                    self.det_latency_ms = det_ms
                    self.detected_faces = faces

                    if faces:
                        crop, coords = self.detector.extract_face_crop(raw_frame, faces[0]['xyxy'], padding=0.12)
                        self.primary_crop = crop
                        self.crop_coords = coords
                    else:
                        self.primary_crop = None
                        self.crop_coords = None

                display_frame = self.render_overlay(raw_frame.copy())
            else:
                display_frame = self.render_overlay(standby_canvas.copy())
                # Standby message in center
                cv2.putText(display_frame, "WEBCAM IS PAUSED / STOPPED", (FRAME_WIDTH // 2 - 200, FRAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (90, 105, 125), 2)
                cv2.putText(display_frame, "Click [> START CAMERA] or press [SPACEBAR]", (FRAME_WIDTH // 2 - 230, FRAME_HEIGHT // 2 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MUTED, 1)

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACEBAR
                self.toggle_camera()
            elif key == ord('+') or key == ord('='):
                self.adjust_confidence(+0.05)
            elif key == ord('-') or key == ord('_'):
                self.adjust_confidence(-0.05)
            elif key == ord('s') or key == ord('S'):
                self.save_snapshot(display_frame)

        # Cleanup
        if self.camera_active:
            self.camera.stop()
        cv2.destroyAllWindows()
        print("\n[Shutdown] Application closed cleanly.")

def main():
    parser = argparse.ArgumentParser(description="SignalWell Standalone YOLO Face Detection Diagnostic Application")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--image", type=str, default="", help="Path to static image for single evaluation")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution (default: 640)")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without opening GUI window")
    args = parser.parse_args()

    app = StandaloneFaceDetectorApp(
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

