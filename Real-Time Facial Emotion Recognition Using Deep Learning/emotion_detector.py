"""
╔══════════════════════════════════════════════════════════════╗
║        REAL-TIME EMOTION DETECTION SYSTEM                   ║
║        Using CNN + OpenCV + TensorFlow/Keras                ║
╚══════════════════════════════════════════════════════════════╝

This is the MAIN entry point for the Real-Time Emotion Detection System.
It ties together:
  - Face detection (Haar Cascade or DNN)
  - Emotion classification (CNN model)
  - Webcam capture (OpenCV)
  - Real-time display with bounding boxes & labels
  - Emotion statistics tracking
  - Alert system for continuous negative emotions
  - CSV logging of detected emotions
"""

import cv2
import numpy as np
import time
import os
import sys
from collections import deque

# ─── Local module imports ─────────────────────────────────────
from utils.face_detector import FaceDetector
from utils.emotion_predictor import EmotionPredictor
from utils.emotion_logger import EmotionLogger
from utils.display_utils import (
    draw_face_box,
    draw_emotion_label,
    draw_stats_panel,
    draw_alert_banner,
    draw_fps_counter,
)
from utils.alert_system import AlertSystem

# ─── Configuration ─────────────────────────────────────────────
WEBCAM_INDEX        = 0          # 0 = default webcam; change if using external camera
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
WINDOW_NAME         = "🎭 Real-Time Emotion Detection  |  Press Q to quit"

# Negative emotion alert: trigger if the same negative emotion appears
# for ALERT_THRESHOLD consecutive seconds
ALERT_THRESHOLD_SEC = 5

# ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Real-Time Emotion Detection System — Starting Up")
    print("="*60)

    # ── 1. Initialise webcam ───────────────────────────────────
    print("[INFO] Opening webcam …")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check WEBCAM_INDEX in emotion_detector.py")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[INFO] Webcam opened at {int(cap.get(3))}×{int(cap.get(4))} px")

    # ── 2. Initialise face detector ────────────────────────────
    print("[INFO] Loading face detector …")
    face_detector = FaceDetector(method="haar")   # "haar" or "dnn"

    # ── 3. Load / build CNN emotion model ─────────────────────
    print("[INFO] Loading emotion recognition model …")
    predictor = EmotionPredictor()

    # ── 4. Emotion logger (CSV) ────────────────────────────────
    logger = EmotionLogger(log_dir="saved_emotions")

    # ── 5. Alert system ───────────────────────────────────────
    alert_system = AlertSystem(threshold_seconds=ALERT_THRESHOLD_SEC)

    # ── 6. FPS tracking ───────────────────────────────────────
    fps_deque   = deque(maxlen=30)          # rolling average over 30 frames
    prev_time   = time.time()

    # ── 7. Emotion statistics counter ─────────────────────────
    emotion_counts = {e: 0 for e in predictor.EMOTIONS}

    print("[INFO] ✅ All systems ready. Press Q inside the window to quit.\n")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ══════════════════════════════════════════════════════════
    #                     MAIN LOOP
    # ══════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame — retrying …")
            continue

        # ── FPS calculation ────────────────────────────────────
        curr_time = time.time()
        fps_deque.append(1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time
        fps       = np.mean(fps_deque)

        # ── Face detection ─────────────────────────────────────
        faces = face_detector.detect(frame)

        current_emotions = []   # emotions detected in this frame

        for (x, y, w, h) in faces:
            # ── Extract & preprocess face ROI ──────────────────
            face_roi = frame[y:y+h, x:x+w]

            # ── Emotion prediction ─────────────────────────────
            emotion_label, confidence, all_probs = predictor.predict(face_roi)

            current_emotions.append(emotion_label)
            emotion_counts[emotion_label] += 1

            # ── Log to CSV ─────────────────────────────────────
            logger.log(emotion_label, confidence)

            # ── Draw bounding box ──────────────────────────────
            draw_face_box(frame, x, y, w, h, emotion_label)

            # ── Draw emotion label + confidence ────────────────
            draw_emotion_label(frame, x, y, emotion_label, confidence)

        # ── Statistics panel (top-right) ──────────────────────
        draw_stats_panel(frame, emotion_counts)

        # ── Alert system check ────────────────────────────────
        alert_msg = alert_system.update(current_emotions)
        if alert_msg:
            draw_alert_banner(frame, alert_msg)

        # ── FPS counter ───────────────────────────────────────
        draw_fps_counter(frame, fps)

        # ── Show frame ────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, frame)

        # ── Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Q pressed — shutting down …")
            break
        elif key == ord('s') or key == ord('S'):
            # Save a screenshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"saved_emotions/screenshot_{ts}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved → {fname}")

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    logger.close()

    # ── Final statistics summary ──────────────────────────────
    print("\n" + "="*60)
    print("  SESSION EMOTION STATISTICS")
    print("="*60)
    total = sum(emotion_counts.values()) or 1
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / total * 30)
        print(f"  {emotion:<12} {bar:<30} {count:>5}  ({count/total*100:.1f}%)")
    print("="*60)
    print(f"  Log saved to: {logger.filepath}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
