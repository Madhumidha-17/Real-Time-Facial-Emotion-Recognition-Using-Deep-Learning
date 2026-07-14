"""
download_pretrained.py
──────────────────────
Downloads a ready-to-use pre-trained emotion model so you don't
need to train from scratch (saves ~2 hours of GPU time).

Two options are provided — the script tries them in order:

  Option A  DeepFace / FER model  (via deepface library)
  Option B  OpenCV DNN emotion model (lightweight, no GPU needed)

After running this script, models/emotion_model.h5 will exist and
emotion_detector.py will load it automatically.

Usage:
    python download_pretrained.py
"""

import os
import sys
import urllib.request
import zipfile

MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")


# ══════════════════════════════════════════════════════════════
#  Option A: Use deepface's bundled FER model
# ══════════════════════════════════════════════════════════════
def try_deepface():
    """
    DeepFace ships its own FER+ emotion model. We just copy its weights
    into our models/ directory so EmotionPredictor can load them.
    """
    try:
        print("[Pretrained] Trying deepface …")
        from deepface import DeepFace
        from deepface.models.facial_attribute import Emotion as EmotionModel
        import tensorflow as tf

        model = EmotionModel.build_model()
        model.save_weights(MODEL_PATH)
        print(f"[Pretrained] ✅  DeepFace emotion weights saved → {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[Pretrained] deepface option failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  Option B: Download from GitHub releases
# ══════════════════════════════════════════════════════════════
# A community-trained FER-2013 Keras model hosted on GitHub.
GITHUB_MODEL_URL = (
    "https://github.com/oarriaga/face_classification/releases/download/"
    "v0.1/fer2013_mini_XCEPTION.102-0.66.hdf5"
)
TEMP_PATH = os.path.join(MODEL_DIR, "_temp_downloaded.hdf5")


def try_github():
    """Download a pre-trained Keras model from GitHub."""
    try:
        print(f"[Pretrained] Downloading model from GitHub …")
        os.makedirs(MODEL_DIR, exist_ok=True)

        def progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / max(total_size, 1))
            print(f"\r  {pct}%  [{pct * '█':<50}]", end="", flush=True)

        urllib.request.urlretrieve(GITHUB_MODEL_URL, TEMP_PATH, reporthook=progress)
        print()

        # This model uses a different architecture — wrap it
        import tensorflow as tf
        from tensorflow import keras

        src_model = keras.models.load_model(TEMP_PATH)
        src_model.save_weights(MODEL_PATH)
        os.remove(TEMP_PATH)

        print(f"[Pretrained] ✅  Model weights saved → {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"\n[Pretrained] GitHub download failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  Option C: Manual instructions
# ══════════════════════════════════════════════════════════════
def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD — pre-trained FER-2013 Keras model             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Visit:                                                       ║
║     https://github.com/oarriaga/face_classification/releases    ║
║                                                                  ║
║  2. Download:                                                    ║
║     fer2013_mini_XCEPTION.102-0.66.hdf5                         ║
║                                                                  ║
║  3. Place it at:                                                 ║
║     models/emotion_model.h5                                     ║
║                                                                  ║
║  4. Then run:                                                    ║
║     python emotion_detector.py                                  ║
║                                                                  ║
║  ─── OR ────────────────────────────────────────────────────── ║
║                                                                  ║
║  Train from scratch with FER-2013 data:                         ║
║     1. Download fer2013.csv from Kaggle                         ║
║        https://www.kaggle.com/datasets/msambare/fer2013         ║
║     2. Place it in  data/fer2013.csv                            ║
║     3. Run:  python train_model.py                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"[Pretrained] Model already exists at {MODEL_PATH} — nothing to do.")
        return

    print("=" * 60)
    print("  Emotion Detection — Pre-trained Model Downloader")
    print("=" * 60)

    if try_deepface():
        return

    if try_github():
        return

    print_manual_instructions()


if __name__ == "__main__":
    main()
