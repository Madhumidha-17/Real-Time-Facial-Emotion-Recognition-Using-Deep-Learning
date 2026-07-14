"""
utils/emotion_logger.py
───────────────────────
Writes every detected emotion + timestamp + confidence to a CSV file
so you can analyse sessions after the fact.

CSV schema:
  timestamp, emotion, confidence
"""

import os
import csv
import time
from datetime import datetime


class EmotionLogger:
    """
    Append-mode CSV logger for detected emotions.

    Parameters
    ----------
    log_dir : str
        Directory where the CSV file is created.
    flush_every : int
        Write buffer to disk every N rows (default: 10).
    """

    FIELDNAMES = ["timestamp", "emotion", "confidence"]

    def __init__(self, log_dir: str = "saved_emotions", flush_every: int = 10):
        os.makedirs(log_dir, exist_ok=True)

        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"emotions_{ts}.csv")
        self._count   = 0
        self._flush_every = flush_every

        self._file   = open(self.filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

        print(f"[EmotionLogger] Logging to {self.filepath}")

    def log(self, emotion: str, confidence: float):
        """Write one row to the CSV."""
        self._writer.writerow({
            "timestamp":  datetime.now().isoformat(timespec="milliseconds"),
            "emotion":    emotion,
            "confidence": f"{confidence:.4f}",
        })
        self._count += 1
        if self._count % self._flush_every == 0:
            self._file.flush()

    def close(self):
        """Flush and close the CSV file."""
        self._file.flush()
        self._file.close()
        print(f"[EmotionLogger] Session log saved → {self.filepath}  ({self._count} entries)")
