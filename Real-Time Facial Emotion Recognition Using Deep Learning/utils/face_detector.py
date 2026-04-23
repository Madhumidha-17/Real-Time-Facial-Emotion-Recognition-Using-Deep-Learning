"""
utils/face_detector.py
──────────────────────
Provides the FaceDetector class with two backends:

  • "haar"  — OpenCV Haar Cascade (fast, works offline, ships with OpenCV)
  • "dnn"   — OpenCV DNN face detector (more accurate, slightly slower)

Usage:
    detector = FaceDetector(method="haar")
    faces = detector.detect(frame)   # returns list of (x, y, w, h) tuples
"""

import cv2
import numpy as np
import os
import urllib.request


# ── Paths ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "..", "models")

HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# DNN model URLs (OpenCV's Caffe face detector)
DNN_PROTO_URL  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL  = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
DNN_PROTO_PATH = os.path.join(_MODELS_DIR, "deploy.prototxt")
DNN_MODEL_PATH = os.path.join(_MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


class FaceDetector:
    """
    Detects human faces in BGR frames.

    Parameters
    ----------
    method : str
        "haar" (default) or "dnn"
    min_confidence : float
        Minimum confidence for DNN detections (ignored for Haar)
    scale_factor : float
        Haar: image scale between successive scans
    min_neighbors : int
        Haar: minimum neighbors each candidate rectangle should retain
    """

    def __init__(
        self,
        method: str = "haar",
        min_confidence: float = 0.5,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
    ):
        self.method         = method.lower()
        self.min_confidence = min_confidence
        self.scale_factor   = scale_factor
        self.min_neighbors  = min_neighbors

        os.makedirs(_MODELS_DIR, exist_ok=True)

        if self.method == "haar":
            self._load_haar()
        elif self.method == "dnn":
            self._load_dnn()
        else:
            raise ValueError(f"Unknown face-detection method: '{method}'. Choose 'haar' or 'dnn'.")

    # ── Haar ──────────────────────────────────────────────────
    def _load_haar(self):
        if not os.path.exists(HAAR_XML):
            raise FileNotFoundError(
                f"Haar cascade XML not found at: {HAAR_XML}\n"
                "Make sure opencv-python is installed correctly."
            )
        self.cascade = cv2.CascadeClassifier(HAAR_XML)
        print(f"[FaceDetector] Haar Cascade loaded ✓")

    # ── DNN ───────────────────────────────────────────────────
    def _load_dnn(self):
        for path, url, name in [
            (DNN_PROTO_PATH, DNN_PROTO_URL,  "deploy.prototxt"),
            (DNN_MODEL_PATH, DNN_MODEL_URL,  "caffemodel"),
        ]:
            if not os.path.exists(path):
                print(f"[FaceDetector] Downloading DNN face detector: {name} …")
                try:
                    urllib.request.urlretrieve(url, path)
                except Exception as e:
                    print(f"[FaceDetector] Download failed ({e}). Falling back to Haar.")
                    self.method = "haar"
                    self._load_haar()
                    return

        self.net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
        print("[FaceDetector] DNN face detector loaded ✓")

    # ── Public API ────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces in a BGR frame.

        Returns
        -------
        List of (x, y, w, h) tuples in pixel coordinates.
        """
        if self.method == "haar":
            return self._detect_haar(frame)
        else:
            return self._detect_dnn(frame)

    # ── Haar detection ────────────────────────────────────────
    def _detect_haar(self, frame: np.ndarray) -> list:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)          # improve contrast

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor  = self.scale_factor,
            minNeighbors = self.min_neighbors,
            minSize      = (48, 48),
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []
        return [(x, y, w, h) for (x, y, w, h) in faces]

    # ── DNN detection ─────────────────────────────────────────
    def _detect_dnn(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        blob  = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor = 1.0,
            size        = (300, 300),
            mean        = (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.min_confidence:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Clip to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            results.append((x1, y1, x2 - x1, y2 - y1))

        return results
