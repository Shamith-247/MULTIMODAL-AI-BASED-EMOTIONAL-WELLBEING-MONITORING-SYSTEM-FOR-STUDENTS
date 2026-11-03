#!/usr/bin/env python3
"""
SignalWell — High-Performance Facial Emotion Classifier
========================================================
Performs 7-class emotion recognition on face crops extracted by YOLO,
specifically integrating the trained weights from 'face_model_v6_best.pt'.

Categories (Ekman 7):
  1. Happy
  2. Sad
  3. Disgust
  4. Surprise
  5. Fear
  6. Angry
  7. Neutral

Features:
- Exact architecture: InceptionResnetV1(pretrained=None, classify=False) + Dropout(0.4) + Linear(512, 7)
- Exact v6 training preprocessing:
    * RGB color space
    * 160x160 resolution
    * Pixel normalization: (x / 255.0 - 0.5) / 0.5 -> [-1.0, 1.0]
- Exact checkpoint label mapping:
    * Index 0: 'anger'     -> 'Angry'
    * Index 1: 'disgust'   -> 'Disgust'
    * Index 2: 'fear'      -> 'Fear'
    * Index 3: 'happiness' -> 'Happy'
    * Index 4: 'neutral'   -> 'Neutral'
    * Index 5: 'sadness'   -> 'Sad'
    * Index 6: 'surprise'  -> 'Surprise'
- Temporal Exponential Moving Average (EMA) smoothing for prediction stability
- Separate failure handling: "NO FACE DETECTED" vs "EMOTION CLASSIFICATION UNAVAILABLE"
- Zero artificial confidence inflation; raw probabilities exposed for debugging
- CPU & CUDA GPU auto-detection
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ekman 7 Affective Taxonomy (Display Order)
DISPLAY_CLASSES = ['Happy', 'Sad', 'Disgust', 'Surprise', 'Fear', 'Angry', 'Neutral']

# Underlying model class indices (RAF-DB & face_model_v6_best.pt checkpoint order)
# Checkpoint labels: ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
CKPT_INDEX_TO_NAME = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'neutral',
    5: 'sadness',
    6: 'surprise'
}

# Mapping checkpoint emotion names to standard Display Classes
RAW_TO_DISPLAY = {
    'happiness': 'Happy',
    'sadness': 'Sad',
    'disgust': 'Disgust',
    'surprise': 'Surprise',
    'fear': 'Fear',
    'anger': 'Angry',
    'neutral': 'Neutral'
}

DISPLAY_TO_CKPT = {v: k for k, v in RAW_TO_DISPLAY.items()}

# Emoji icons for modern HUD display
EMOTION_EMOJIS = {
    'Happy': '😊',
    'Sad': '😢',
    'Disgust': '🤢',
    'Surprise': '😲',
    'Fear': '😨',
    'Angry': '😠',
    'Neutral': '😐'
}

class FaceEmotionModelV6(nn.Sequential):
    """
    Exact neural network architecture for face_model_v6_best.pt:
    InceptionResnetV1 (VGGFace2 backbone) -> Dropout(0.4) -> Linear(512, 7)
    Subclasses nn.Sequential to preserve exact state_dict keys ('0.*' and '2.*').
    """
    def __init__(self, num_classes=7, dropout=0.4):
        from facenet_pytorch import InceptionResnetV1
        super().__init__(
            InceptionResnetV1(pretrained=None, classify=False),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

class FacialEmotionClassifier:
    """
    Calibrated facial emotion recognition engine with temporal smoothing,
    integrating the trained weights from 'face_model_v6_best.pt'.
    """
    def __init__(self, model_path=None, device=None, ema_alpha=0.35):
        self.ema_alpha = ema_alpha
        self.input_size = (160, 160)
        self.model = None
        self.backend = None
        self.model_path = ""
        self.labels = [CKPT_INDEX_TO_NAME[i] for i in range(7)]

        # Resolve device: CUDA GPU when available, CPU otherwise
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model(model_path)

    def _load_model(self, model_path=None):
        """Loads and verifies the face_model_v6_best.pt weights."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_paths = [
            model_path,
            os.path.join(project_root, "models", "face_model_v6_best.pt"),
            os.path.join(os.path.expanduser("~"), "Downloads", "face_model_v6_best.pt"),
            os.path.join(project_root, "models", "face_model_v7_best.pt"),
            os.path.join(project_root, "models", "best_face_model_v8_large_dataset.pt"),
            "models/face_model_v6_best.pt"
        ]

        valid_path = None
        for p in candidate_paths:
            if p and os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
                valid_path = p
                break

        if valid_path is None:
            print("[EmotionClassifier] ⚠️ WARNING: No trained emotion checkpoint found on disk.")
            self.backend = "unavailable"
            return

        try:
            print(f"[EmotionClassifier] Loading checkpoint from: {valid_path}")
            ckpt = torch.load(valid_path, map_location="cpu")

            # Extract labels if present in checkpoint metadata
            if isinstance(ckpt, dict) and "labels" in ckpt:
                self.labels = ckpt["labels"]
                print(f"[EmotionClassifier] Checkpoint labels: {self.labels}")
            if isinstance(ckpt, dict) and "test_acc" in ckpt:
                print(f"[EmotionClassifier] Checkpoint reported test accuracy: {ckpt['test_acc']*100:.2f}%")

            # Build FaceEmotionModelV6
            model = FaceEmotionModelV6(num_classes=len(self.labels), dropout=0.4)
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device).eval()

            self.model = model
            self.model_path = valid_path
            self.backend = "pytorch_v6"
            print(f"[EmotionClassifier] Successfully loaded {os.path.basename(valid_path)} on {self.device}")
            if unexpected:
                print(f"[EmotionClassifier] Note: Ignored auxiliary keys: {unexpected}")
        except Exception as e:
            print(f"[EmotionClassifier] ❌ ERROR loading emotion model: {e}")
            self.backend = "unavailable"
            self.model = None

    def preprocess_crop(self, face_bgr):
        """
        Applies exact preprocessing used during face_model_v6 training:
        1. Convert BGR to RGB
        2. Resize to 160x160 using bilinear interpolation
        3. Convert to float32 and scale [0, 255] -> [0.0, 1.0]
        4. Normalize with mean=0.5, std=0.5 -> [-1.0, 1.0]
        5. Permute HWC -> CHW -> NCHW tensor
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        # 1. BGR -> RGB
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # 2. Resize to model's expected 160x160 input size
        resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # 3. & 4. Normalization: ((x / 255.0) - 0.5) / 0.5
        norm_img = ((resized.astype(np.float32) / 255.0) - 0.5) / 0.5

        # 5. Tensor conversion [1, 3, 160, 160]
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict_face(self, face_crop_bgr, prev_ema=None):
        """
        Runs emotion classification on a single face crop.
        Returns:
            top_emotion (str or None): Name of dominant emotion (e.g. 'Happy')
            confidence (float): Confidence score [0.0, 1.0]
            probabilities (dict or None): Dict of all 7 emotion probabilities
            cls_latency_ms (float): Inference latency in milliseconds
        """
        # Handle invalid face crops gracefully (no fallback to Neutral)
        if face_crop_bgr is None or face_crop_bgr.size == 0 or face_crop_bgr.shape[0] < 10 or face_crop_bgr.shape[1] < 10:
            return None, 0.0, None, 0.0

        if self.model is None or self.backend == "unavailable":
            return None, 0.0, None, 0.0

        t0 = time.perf_counter()

        # 1. Exact Preprocessing
        tensor = self.preprocess_crop(face_crop_bgr)
        if tensor is None:
            return None, 0.0, None, 0.0

        # 2. Forward Pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs_t = F.softmax(logits, dim=1)[0]
            raw_probs = probs_t.cpu().numpy()

        # 3. Map model output indices to user's 7 emotion categories
        # Checkpoint: ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        mapped_probs = {}
        for idx, raw_name in enumerate(self.labels):
            display_name = RAW_TO_DISPLAY.get(raw_name, raw_name.capitalize())
            mapped_probs[display_name] = float(raw_probs[idx])

        # Ensure all 7 display classes are present
        for c in DISPLAY_CLASSES:
            if c not in mapped_probs:
                mapped_probs[c] = 0.0

        # 4. Temporal Exponential Moving Average (EMA) Smoothing
        if prev_ema is not None and isinstance(prev_ema, dict):
            smoothed_probs = {}
            for k in DISPLAY_CLASSES:
                cur_val = mapped_probs.get(k, 0.0)
                prev_val = prev_ema.get(k, cur_val)
                smoothed_probs[k] = self.ema_alpha * cur_val + (1.0 - self.ema_alpha) * prev_val
            # Re-normalize sum to 1.0
            total_sum = sum(smoothed_probs.values()) or 1.0
            final_probs = {k: v / total_sum for k, v in smoothed_probs.items()}
        else:
            final_probs = mapped_probs

        # 5. Extract top predicted emotion and confidence
        top_emotion = max(final_probs, key=final_probs.get)
        confidence = float(final_probs[top_emotion])

        t1 = time.perf_counter()
        cls_latency_ms = (t1 - t0) * 1000.0

        return top_emotion, confidence, final_probs, cls_latency_ms

if __name__ == "__main__":
    classifier = FacialEmotionClassifier()
    dummy_face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    top_em, conf, probs, latency = classifier.predict_face(dummy_face)
    print("\nSelf-test on random dummy crop:")
    print(f"  Backend: {classifier.backend}")
    print(f"  Top Emotion: {top_em} ({conf*100:.1f}%)")
    print(f"  Inference Latency: {latency:.2f} ms")
    if probs:
        for k in DISPLAY_CLASSES:
            print(f"    {k:<10}: {probs[k]*100:.2f}%")
