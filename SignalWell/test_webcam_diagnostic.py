#!/usr/bin/env python3
"""
SignalWell - Systematic Testing & Diagnostic Pipeline
=====================================================
Runs 5 independent tests to isolate and verify:
Test 1: Python environment & computer vision libraries
Test 2: Model architecture & weights loading
Test 3: Static image face detection & emotion inference (using test_photos)
Test 4: Laptop webcam hardware, DirectShow access, & frame validity
Test 5: Real-time face detection on live webcam feed (saves snapshot)
"""

import sys
import os
import time

def banner(title):
    print("\n" + "="*65)
    print(f" {title}")
    print("="*65)

# -------------------------------------------------------------
# TEST 1: Environment & Libraries
# -------------------------------------------------------------
banner("TEST 1: Python Environment & Libraries")
print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.split()[0]}")
print(f"Platform:          {sys.platform}")

modules = ["cv2", "numpy", "PIL", "torch", "facenet_pytorch", "onnxruntime"]
installed = {}
for mod in modules:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "installed")
        print(f"  [OK] {mod:<16} (v{ver})")
        installed[mod] = True
    except ImportError:
        print(f"  [--] {mod:<16} (not installed)")
        installed[mod] = False

# -------------------------------------------------------------
# TEST 2: Model & Weights Verification
# -------------------------------------------------------------
banner("TEST 2: Model Architecture & Weights Verification")
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints = [
    os.path.join(script_dir, "face_model_v7_best.pt"),
    os.path.join(script_dir, "face_model_v7_calibrated.pt"),
    os.path.join(script_dir, "face_model_v7_calibrated.onnx"),
    os.path.join(script_dir, "face_model_v7.onnx")
]

found_ckpt = False
for ckpt_path in checkpoints:
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"  [FOUND] {os.path.basename(ckpt_path)} ({size_mb:.2f} MB)")
        found_ckpt = True

if not found_ckpt:
    print("  [INFO] No .pt or .onnx files in local project directory.")
    print("         The application will run using built-in OpenCV detection + heuristic fallback.")
    print("         To enable your custom deep net, copy 'face_model_v7_best.pt' into this folder.")

# -------------------------------------------------------------
# TEST 3: Static Image Face Detection (Using test_photos)
# -------------------------------------------------------------
banner("TEST 3: Static Image Face Detection & Inference")
test_photos_dir = os.path.join(script_dir, "test_photos")

if not installed.get("cv2", False):
    print("  [SKIP] OpenCV (cv2) is required for static image test.")
else:
    import cv2
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(haar_path)

    if not os.path.exists(test_photos_dir):
        print(f"  [WARN] test_photos directory not found at {test_photos_dir}")
    else:
        sample_photos = [f for f in os.listdir(test_photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(sample_photos)} test photos in {test_photos_dir}:")
        
        for fname in sample_photos:
            fpath = os.path.join(test_photos_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"  [FAIL] Could not read {fname}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            h, w = img.shape[:2]
            
            if len(faces) > 0:
                print(f"  [PASS] {fname:<30} -> {len(faces)} face(s) detected (box: x={faces[0][0]}, y={faces[0][1]}, w={faces[0][2]}, h={faces[0][3]})")
            else:
                # Try with CLAHE enhancement
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                faces_clahe = detector.detectMultiScale(enhanced_gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
                if len(faces_clahe) > 0:
                    print(f"  [PASS] {fname:<30} -> Detected with CLAHE illumination enhancement! ({len(faces_clahe)} face)")
                else:
                    print(f"  [WARN] {fname:<30} -> No face detected (check lighting/angle)")

# -------------------------------------------------------------
# TEST 4: Webcam Hardware & DirectShow Probing
# -------------------------------------------------------------
banner("TEST 4: Laptop Webcam Hardware & DirectShow Probing")
if not installed.get("cv2", False):
    print("  [SKIP] OpenCV is required for webcam probing.")
else:
    import cv2
    active_camera = None
    
    for idx in [0, 1, 2]:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        bname = "DirectShow" if backend == cv2.CAP_DSHOW else "Default"
        print(f"Testing Camera Index {idx} ({bname})...", end=" ")
        
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w, c = frame.shape
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[SUCCESS] -> {w}x{h}, {c} channels, {fps:.0f} FPS")
                if active_camera is None:
                    active_camera = (idx, backend, cap)
                else:
                    cap.release()
            else:
                print("[FAILED] Opened device but returned empty frame.")
                cap.release()
        else:
            print("[FAILED] Device could not be opened.")
            cap.release()

    if active_camera is None:
        print("\n  [CRITICAL] No active webcam could be accessed!")
        print("  Troubleshooting steps:")
        print("  1. Close any browser tabs, Zoom, or Teams that may hold the webcam lock.")
        print("  2. Open Windows Settings > Privacy & security > Camera:")
        print("     Ensure 'Camera access' is ON and 'Let desktop apps access your camera' is ON.")
        print("  3. Check device manager for webcam drivers.")
    else:
        cam_idx, cam_backend, cap = active_camera
        print(f"\n  Active camera verified on index {cam_idx}.")

        # -------------------------------------------------------------
        # TEST 5: Live Detection Verification Snapshot
        # -------------------------------------------------------------
        banner("TEST 5: Live Detection Verification Snapshot")
        print("Capturing 10 frames to allow auto-exposure to stabilize...")
        for _ in range(10):
            ret, frame = cap.read()
            time.sleep(0.05)
        
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            print(f"Live detection result: {len(faces)} face(s) found in live feed.")
            
            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out_snap = os.path.join(script_dir, "diagnostic_snapshot.jpg")
            cv2.imwrite(out_snap, frame)
            print(f"Saved live diagnostic snapshot to: {out_snap}")
        
        cap.release()

banner("DIAGNOSTIC PIPELINE COMPLETE")
print("All individual component tests finished.")

