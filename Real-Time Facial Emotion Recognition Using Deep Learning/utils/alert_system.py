"""
utils/alert_system.py
─────────────────────
Fires an alert when a NEGATIVE emotion is detected continuously
for longer than `threshold_seconds`.

Negative emotions tracked: Angry, Disgust, Fear, Sad
"""

import time


NEGATIVE_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad"}

ALERT_MESSAGES = {
    "Angry":   "⚠  ALERT: Continuous ANGER detected!",
    "Disgust": "⚠  ALERT: Sustained DISGUST detected!",
    "Fear":    "⚠  ALERT: Prolonged FEAR detected!",
    "Sad":     "⚠  ALERT: Continuous SADNESS detected!",
}


class AlertSystem:
    """
    Tracks the dominant emotion frame-by-frame and raises an alert
    if a negative emotion persists beyond the threshold.

    Parameters
    ----------
    threshold_seconds : float
        How long (in wall-clock seconds) a negative emotion must persist
        before an alert is returned.
    cooldown_seconds : float
        After an alert fires, wait this long before firing again.
    """

    def __init__(self, threshold_seconds: float = 5.0, cooldown_seconds: float = 10.0):
        self.threshold  = threshold_seconds
        self.cooldown   = cooldown_seconds

        self._streak_emotion = None
        self._streak_start   = None
        self._last_alert     = 0.0          # timestamp of last alert

    def update(self, detected_emotions: list) -> str | None:
        """
        Call once per frame with the list of emotions detected in that frame.

        Returns
        -------
        Alert message string if threshold crossed, else None.
        """
        now = time.time()

        # Pick the "dominant" emotion for this frame (first face, or None)
        dominant = detected_emotions[0] if detected_emotions else None

        if dominant in NEGATIVE_EMOTIONS:
            if self._streak_emotion == dominant:
                # Streak continues
                elapsed = now - self._streak_start
                if elapsed >= self.threshold:
                    if (now - self._last_alert) >= self.cooldown:
                        self._last_alert = now
                        return ALERT_MESSAGES.get(dominant, f"⚠  Negative emotion: {dominant}")
            else:
                # New negative emotion — start fresh streak
                self._streak_emotion = dominant
                self._streak_start   = now
        else:
            # Positive / neutral emotion — reset streak
            self._streak_emotion = None
            self._streak_start   = None

        return None
