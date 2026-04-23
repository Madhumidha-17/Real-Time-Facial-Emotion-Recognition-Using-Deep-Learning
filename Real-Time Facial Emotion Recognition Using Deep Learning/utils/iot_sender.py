"""
utils/iot_sender.py
───────────────────
Optional IoT integration: send emotion data to ThingSpeak
(a free IoT analytics cloud platform).

Setup
-----
1. Create a free account at https://thingspeak.com
2. Create a channel with 8 fields:
     Field 1: Angry
     Field 2: Disgust
     Field 3: Fear
     Field 4: Happy
     Field 5: Neutral
     Field 6: Sad
     Field 7: Surprise
     Field 8: Dominant emotion (encoded as index 0-6)
3. Copy your channel's WRITE API KEY
4. Set environment variable or edit WRITE_API_KEY below

Usage (inside emotion_detector.py):
    from utils.iot_sender import ThingSpeakSender
    sender = ThingSpeakSender(api_key="YOUR_KEY")
    sender.send(emotion_counts, dominant_emotion)
"""

import os
import time
import threading
import urllib.request
import urllib.parse
from typing import Dict


THINGSPEAK_URL = "https://api.thingspeak.com/update"
EMOTIONS       = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
MIN_INTERVAL   = 15   # ThingSpeak free tier: max 1 update per 15 s


class ThingSpeakSender:
    """
    Non-blocking background thread that sends emotion counts to ThingSpeak.

    Parameters
    ----------
    api_key : str
        ThingSpeak Write API Key.  If not provided, reads
        the THINGSPEAK_API_KEY environment variable.
    interval : int
        Seconds between uploads (minimum 15 for free tier).
    """

    def __init__(self, api_key: str = "", interval: int = 15):
        self.api_key  = api_key or os.getenv("THINGSPEAK_API_KEY", "")
        self.interval = max(interval, MIN_INTERVAL)
        self._pending = None
        self._lock    = threading.Lock()
        self._last_sent = 0.0
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._running = True
        self._thread.start()

        if not self.api_key:
            print("[IoT] ⚠  No ThingSpeak API key provided — IoT sending disabled.")
        else:
            print(f"[IoT] ThingSpeak sender initialised (interval={self.interval}s)")

    def send(self, emotion_counts: Dict[str, int], dominant_emotion: str):
        """Queue a new payload (non-blocking, discards if a send is pending)."""
        if not self.api_key:
            return
        with self._lock:
            self._pending = (dict(emotion_counts), dominant_emotion)

    def _build_payload(self, counts: Dict[str, int], dominant: str) -> str:
        params = {"api_key": self.api_key}
        for i, emotion in enumerate(EMOTIONS, start=1):
            params[f"field{i}"] = counts.get(emotion, 0)
        params["field8"] = EMOTIONS.index(dominant) if dominant in EMOTIONS else -1
        return urllib.parse.urlencode(params)

    def _worker(self):
        while self._running:
            time.sleep(1)
            now = time.time()
            if now - self._last_sent < self.interval:
                continue
            with self._lock:
                payload = self._pending
                self._pending = None

            if payload is None:
                continue

            counts, dominant = payload
            try:
                url    = f"{THINGSPEAK_URL}?{self._build_payload(counts, dominant)}"
                req    = urllib.request.Request(url)
                resp   = urllib.request.urlopen(req, timeout=5)
                result = resp.read().decode()
                if result != "0":
                    print(f"[IoT] ✅  ThingSpeak updated — entry #{result}")
                else:
                    print("[IoT] ⚠  ThingSpeak returned 0 (rate limit or bad key)")
                self._last_sent = now
            except Exception as e:
                print(f"[IoT] ❌  Send failed: {e}")

    def stop(self):
        self._running = False
