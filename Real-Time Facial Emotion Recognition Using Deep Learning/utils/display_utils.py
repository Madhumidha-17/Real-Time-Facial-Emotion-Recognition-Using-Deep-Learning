"""
utils/display_utils.py
──────────────────────
All OpenCV drawing helpers used by emotion_detector.py:

  draw_face_box       — coloured bounding rectangle around detected face
  draw_emotion_label  — emotion label + confidence bar below/above the box
  draw_stats_panel    — emotion count histogram in the top-right corner
  draw_alert_banner   — full-width red alert ribbon at the top
  draw_fps_counter    — FPS counter in the bottom-left corner
"""

import cv2
import numpy as np
from typing import Dict


# ── Typography helpers ────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def _text_size(text: str, font=FONT, scale=0.6, thickness=1):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    return w, h, baseline


# ══════════════════════════════════════════════════════════════
#  Face bounding box
# ══════════════════════════════════════════════════════════════
def draw_face_box(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    emotion_label: str,
    colour_map: dict = None,
):
    """
    Draw a coloured rounded-corner-style bounding box around a face.
    The corner brackets give a more modern "scanner" look than a plain rect.
    """
    # ── Emotion colours (BGR) ─────────────────────────────────
    DEFAULT_COLOURS = {
        "Angry":    (0,   0,   220),
        "Disgust":  (0,   140, 0  ),
        "Fear":     (130, 0,   130),
        "Happy":    (0,   210, 0  ),
        "Neutral":  (180, 180, 180),
        "Sad":      (220, 100, 0  ),
        "Surprise": (0,   200, 255),
    }
    colours = colour_map or DEFAULT_COLOURS
    colour  = colours.get(emotion_label, (200, 200, 200))

    corner  = min(w, h) // 6          # length of each corner bracket
    thick   = 3

    # Draw corner brackets (top-left, top-right, bottom-left, bottom-right)
    pts = [
        # top-left
        [(x, y + corner), (x, y), (x + corner, y)],
        # top-right
        [(x+w - corner, y), (x+w, y), (x+w, y + corner)],
        # bottom-left
        [(x, y+h - corner), (x, y+h), (x + corner, y+h)],
        # bottom-right
        [(x+w - corner, y+h), (x+w, y+h), (x+w, y+h - corner)],
    ]
    for bracket in pts:
        for i in range(len(bracket) - 1):
            cv2.line(frame, bracket[i], bracket[i+1], colour, thick, cv2.LINE_AA)

    # Subtle semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), colour, -1)
    cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)


# ══════════════════════════════════════════════════════════════
#  Emotion label + confidence
# ══════════════════════════════════════════════════════════════
_EMOJI_MAP = {
    "Angry":    "ANGRY",
    "Disgust":  "DISGUST",
    "Fear":     "FEAR",
    "Happy":    "HAPPY",
    "Neutral":  "NEUTRAL",
    "Sad":      "SAD",
    "Surprise": "SURPRISE",
}

def draw_emotion_label(
    frame: np.ndarray,
    x: int, y: int,
    emotion_label: str,
    confidence: float,
    colour_map: dict = None,
):
    """
    Draw pill-shaped label with emotion name + confidence bar
    just above the bounding box.
    """
    DEFAULT_COLOURS = {
        "Angry":    (0,   0,   220),
        "Disgust":  (0,   140, 0  ),
        "Fear":     (130, 0,   130),
        "Happy":    (0,   210, 0  ),
        "Neutral":  (180, 180, 180),
        "Sad":      (220, 100, 0  ),
        "Surprise": (0,   200, 255),
    }
    colours = colour_map or DEFAULT_COLOURS
    colour  = colours.get(emotion_label, (200, 200, 200))

    label_text = f"{emotion_label.upper()}  {confidence*100:.0f}%"
    scale      = 0.65
    thick      = 2
    tw, th, _  = _text_size(label_text, FONT, scale, thick)

    pad    = 6
    box_x1 = x
    box_y1 = max(y - th - pad * 2, 0)
    box_x2 = x + tw + pad * 2
    box_y2 = y

    # Filled background pill
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), colour, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Text
    cv2.putText(
        frame, label_text,
        (box_x1 + pad, box_y2 - pad),
        FONT, scale,
        (255, 255, 255),
        thick, cv2.LINE_AA,
    )

    # ── Mini confidence bar ───────────────────────────────────
    bar_y  = box_y2 + 4
    bar_x1 = x
    bar_w  = int((box_x2 - box_x1) * confidence)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + (box_x2 - box_x1), bar_y + 4),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + 4),
                  colour, -1)


# ══════════════════════════════════════════════════════════════
#  Statistics panel
# ══════════════════════════════════════════════════════════════
def draw_stats_panel(
    frame: np.ndarray,
    emotion_counts: Dict[str, int],
):
    """
    Draw a compact emotion-count histogram in the top-right corner.
    """
    h_frame, w_frame = frame.shape[:2]

    PANEL_W   = 230
    PANEL_H   = 30 + len(emotion_counts) * 28 + 10
    PANEL_X   = w_frame - PANEL_W - 12
    PANEL_Y   = 12

    COLOURS = {
        "Angry":    (0,   0,   220),
        "Disgust":  (0,   140, 0  ),
        "Fear":     (130, 0,   130),
        "Happy":    (0,   210, 0  ),
        "Neutral":  (180, 180, 180),
        "Sad":      (220, 100, 0  ),
        "Surprise": (0,   200, 255),
    }

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (PANEL_X - 8, PANEL_Y - 8),
                  (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Header
    cv2.putText(frame, "EMOTION STATS",
                (PANEL_X, PANEL_Y + 14),
                FONT_SMALL, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    total = max(sum(emotion_counts.values()), 1)
    max_w = PANEL_W - 80

    for i, (emotion, count) in enumerate(emotion_counts.items()):
        row_y  = PANEL_Y + 34 + i * 28
        colour = COLOURS.get(emotion, (200, 200, 200))
        ratio  = count / total
        bar_w  = int(ratio * max_w)

        # Background track
        cv2.rectangle(frame,
                      (PANEL_X, row_y),
                      (PANEL_X + max_w, row_y + 16),
                      (50, 50, 50), -1)
        # Filled bar
        if bar_w > 0:
            cv2.rectangle(frame,
                          (PANEL_X, row_y),
                          (PANEL_X + bar_w, row_y + 16),
                          colour, -1)

        # Label
        cv2.putText(frame, f"{emotion:<8} {count:>4}",
                    (PANEL_X + max_w + 6, row_y + 13),
                    FONT_SMALL, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  Alert banner
# ══════════════════════════════════════════════════════════════
def draw_alert_banner(frame: np.ndarray, message: str):
    """
    Draw a red full-width banner at the top of the frame.
    """
    h_frame, w_frame = frame.shape[:2]
    BANNER_H = 44

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, BANNER_H), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    tw, th, _ = _text_size(message, FONT, 0.65, 2)
    tx = (w_frame - tw) // 2
    ty = (BANNER_H + th) // 2

    cv2.putText(frame, message, (tx, ty),
                FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  FPS counter
# ══════════════════════════════════════════════════════════════
def draw_fps_counter(frame: np.ndarray, fps: float):
    """
    Draw FPS in the bottom-left corner.
    """
    h, _ = frame.shape[:2]
    text  = f"FPS: {fps:.1f}"
    cv2.putText(frame, text,
                (12, h - 12),
                FONT_SMALL, 0.55, (0, 220, 120), 2, cv2.LINE_AA)
