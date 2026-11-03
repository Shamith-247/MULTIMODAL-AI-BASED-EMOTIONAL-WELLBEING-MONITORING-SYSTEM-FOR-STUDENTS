#!/usr/bin/env python3
"""
SignalWell — Integrated YOLO Face Detection & Emotion Recognition Web Server
=============================================================================
Serves the web client and provides high-speed HTTP API endpoints for:
  - Real-time webcam frame inference (YOLO face detection + emotion classification)
  - Sample photos benchmarking (runs YOLO + face_model_v6_best.pt on static photos)
  - Session evaluation management (1-minute / 2-minute session timer & reporting)

Zero external web frameworks required (uses standard library http.server.ThreadingHTTPServer).
"""

import os
import sys
import json
import base64
import time
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

import cv2
import numpy as np

# Ensure root paths resolve cleanly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
APP_DIR = os.path.join(SCRIPT_DIR, "app")

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
from session_evaluator import SessionEvaluator

# Initialize Models Once at Server Startup
print("\n" + "=" * 66)
print(" SIGNALWELL -- INITIALIZING WEB INFERENCE ENGINE")
print("=" * 66)

print("[Server] Loading YOLO Face Detector (yolov11n-face.pt)...")
detector = YOLOFaceDetector(conf_thresh=0.35, imgsz=640)

print("[Server] Loading Emotion Classifier (face_model_v6_best.pt)...")
classifier = FacialEmotionClassifier(ema_alpha=0.35)

evaluator = SessionEvaluator(selected_duration_sec=60)
face_ema_tracker = {}

print("[Server] Neural models loaded and ready.\n")

class IntegratedWebHandler(SimpleHTTPRequestHandler):
    """
    Serves static frontend assets from SignalWell/app/ and handles API endpoints.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=APP_DIR, **kwargs)

    def do_GET(self):
        url = urlparse(self.path)

        if url.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/test_yolo_emotion_web.html")
            self.end_headers()
            return

        if url.path == "/api/status":
            data = {
                "status": "online",
                "face_detector": detector.model_name,
                "emotion_classifier": "face_model_v6_best.pt",
                "backend": classifier.backend,
                "device": classifier.device,
                "classes": DISPLAY_CLASSES,
                "session_state": evaluator.state,
                "session_duration": evaluator.selected_duration_sec
            }
            self._send_json(data)
            return

        if url.path == "/api/session/report":
            report = evaluator.completed_report or evaluator.finalize_session()
            self._send_json(report)
            return

        if url.path == "/api/session/status":
            elapsed = evaluator.get_elapsed_seconds()
            data = {
                "state": evaluator.state,
                "sessionId": evaluator.session_id,
                "selectedDuration": evaluator.selected_duration_sec,
                "elapsedSeconds": round(elapsed, 1),
                "remainingSeconds": round(evaluator.get_remaining_seconds(), 1),
                "samplesCount": len(evaluator.samples),
                "isTimeUp": evaluator.is_time_up()
            }
            self._send_json(data)
            return

        # Fallback to standard static file server
        super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)

        # Read JSON request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        payload = {}
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, str):
                    payload = json.loads(payload)
            except Exception:
                payload = {}

        if url.path == "/api/predict_frame":
            self._handle_predict_frame(payload)
            return

        if url.path == "/api/predict_sample":
            self._handle_predict_sample(payload)
            return

        if url.path == "/api/session/start":
            dur = int(payload.get("duration", 60))
            evaluator.start_session(duration_sec=dur)
            face_ema_tracker.clear()
            self._send_json({"status": "started", "sessionId": evaluator.session_id, "duration": dur})
            return

        if url.path == "/api/session/stop":
            report = evaluator.finalize_session()
            json_file = evaluator.export_json()
            csv_file = evaluator.export_csv()
            html_file = evaluator.export_html_dashboard()
            self._send_json({
                "status": "stopped",
                "report": report,
                "exported": {
                    "json": os.path.basename(json_file),
                    "csv": os.path.basename(csv_file),
                    "html": os.path.basename(html_file)
                }
            })
            return

        if url.path == "/api/session/reset":
            evaluator.reset_session()
            face_ema_tracker.clear()
            self._send_json({"status": "reset"})
            return

        self.send_error(404, "Endpoint not found")

    def _handle_predict_frame(self, payload):
        """Processes a live webcam frame sent as base64 JPEG from the browser."""
        img_b64 = payload.get("image", "")
        if not img_b64:
            self._send_json({"error": "No image data provided"}, status=400)
            return

        # Strip header if present
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            self._send_json({"error": f"Invalid image encoding: {e}"}, status=400)
            return

        if frame is None:
            self._send_json({"error": "Failed to decode image"}, status=400)
            return

        t0 = time.perf_counter()

        # 1. YOLO Face Detection
        faces, det_ms = detector.detect(frame)

        results = []
        cls_total_ms = 0.0

        if faces:
            for idx, f in enumerate(faces):
                xyxy = f["xyxy"]
                # 2. Extract clamped, 12% padded face crop
                crop, coords = detector.extract_face_crop(frame, xyxy, padding=0.12)
                if crop is not None:
                    # 3. Predict emotion with face_model_v6_best.pt
                    prev_ema = face_ema_tracker.get(idx)
                    top_em, conf, probs, cls_ms = classifier.predict_face(crop, prev_ema)
                    cls_total_ms += cls_ms
                    face_ema_tracker[idx] = probs

                    # Encode thumbnail for PiP inset
                    _, thumb_buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_buf).decode("ascii")

                    # Record to active session if running
                    if idx == 0 and evaluator.state == "RUNNING":
                        evaluator.record_frame(
                            has_face=True,
                            raw_emotion=top_em,
                            raw_conf=conf,
                            smoothed_emotion=top_em,
                            probabilities=probs
                        )

                    results.append({
                        "box": xyxy,
                        "det_conf": round(f["conf"], 3),
                        "emotion": top_em,
                        "confidence": round(conf, 3),
                        "emoji": EMOTION_EMOJIS.get(top_em, ""),
                        "probabilities": {k: round(v, 4) for k, v in probs.items()},
                        "crop_thumb": thumb_b64
                    })
                else:
                    # Crop failed
                    if idx == 0 and evaluator.state == "RUNNING":
                        evaluator.record_frame(has_face=True, cls_error=True)

                    results.append({
                        "box": xyxy,
                        "det_conf": round(f["conf"], 3),
                        "emotion": None,
                        "confidence": 0.0,
                        "emoji": "",
                        "probabilities": None,
                        "crop_thumb": ""
                    })
        else:
            # No face detected
            if evaluator.state == "RUNNING":
                evaluator.record_frame(has_face=False)
            face_ema_tracker.clear()

        # Check for auto-finalization if timer expired
        is_completed = False
        if evaluator.is_time_up():
            evaluator.finalize_session()
            evaluator.export_json()
            evaluator.export_csv()
            evaluator.export_html_dashboard()
            is_completed = True

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        resp = {
            "has_face": len(faces) > 0,
            "face_count": len(faces),
            "faces": results,
            "det_latency_ms": round(det_ms, 1),
            "cls_latency_ms": round(cls_total_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "session": {
                "state": evaluator.state,
                "elapsed": round(evaluator.get_elapsed_seconds(), 1),
                "remaining": round(evaluator.get_remaining_seconds(), 1),
                "duration": evaluator.selected_duration_sec,
                "samples": len(evaluator.samples),
                "is_completed": is_completed
            }
        }
        self._send_json(resp)

    def _handle_predict_sample(self, payload):
        """Processes one of the 4 authentic sample test photos."""
        filename = payload.get("filename", "")
        # Resolve path
        candidates = [
            os.path.join(APP_DIR, filename),
            os.path.join(APP_DIR, "test_photos", os.path.basename(filename)),
            os.path.join(SCRIPT_DIR, "test_photos", os.path.basename(filename))
        ]
        valid_path = None
        for c in candidates:
            if os.path.exists(c):
                valid_path = c
                break

        if not valid_path:
            self._send_json({"error": f"Sample photo not found: {filename}"}, status=404)
            return

        frame = cv2.imread(valid_path)
        if frame is None:
            self._send_json({"error": "Failed to decode sample image"}, status=500)
            return

        h, w = frame.shape[:2]
        faces, det_ms = detector.detect(frame)

        results = []
        cls_total = 0.0

        for f in faces:
            xyxy = f["xyxy"]
            crop, _ = detector.extract_face_crop(frame, xyxy, padding=0.12)
            if crop is not None:
                top_em, conf, probs, cls_ms = classifier.predict_face(crop)
                cls_total += cls_ms
                _, thumb_buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_buf).decode("ascii")

                results.append({
                    "box": xyxy,
                    "det_conf": round(f["conf"], 3),
                    "emotion": top_em,
                    "confidence": round(conf, 3),
                    "emoji": EMOTION_EMOJIS.get(top_em, ""),
                    "probabilities": {k: round(v, 4) for k, v in probs.items()},
                    "crop_thumb": thumb_b64
                })

        resp = {
            "filename": os.path.basename(valid_path),
            "width": w,
            "height": h,
            "has_face": len(faces) > 0,
            "faces": results,
            "det_latency_ms": round(det_ms, 1),
            "cls_latency_ms": round(cls_total, 1),
            "total_latency_ms": round(det_ms + cls_total, 1)
        }
        self._send_json(resp)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def run_server(port=8000):
    server = ThreadingHTTPServer(("0.0.0.0", port), IntegratedWebHandler)
    print(f"\n[OK] SignalWell Integrated YOLO + Emotion Web Server running at:")
    print(f"     >> http://localhost:{port}/test_yolo_emotion_web.html")
    print(f"     >> http://127.0.0.1:{port}/test_yolo_emotion_web.html\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.server_close()

if __name__ == "__main__":
    run_server(8000)
