"""
train_model.py
──────────────
Train the EmotionCNN from scratch on the FER-2013 dataset.

Quick-start
-----------
1. Download FER-2013 from https://www.kaggle.com/datasets/msambare/fer2013
2. Place the CSV at:   data/fer2013.csv
3. Run:                python train_model.py

The trained weights will be saved to models/emotion_model.h5

Training config (edit as needed):
  EPOCHS       = 60
  BATCH_SIZE   = 64
  LR           = 1e-3
  DATA_AUG     = True   (random flips, shifts, zoom, rotation)
"""

import os
import sys
import numpy as np

# ── Ensure TensorFlow is available ────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, callbacks
    print(f"[Train] TensorFlow {tf.__version__} ✓")
except ImportError:
    print("[Train] TensorFlow not found. Install with:  pip install tensorflow")
    sys.exit(1)

import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────
DATA_CSV    = os.path.join("data", "fer2013.csv")
MODEL_PATH  = os.path.join("models", "emotion_model.h5")
LOG_DIR     = os.path.join("logs", "training")

IMG_SIZE    = 48
NUM_CLASSES = 7
EPOCHS      = 60
BATCH_SIZE  = 64
LR          = 1e-3

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


# ══════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════
def load_fer2013(csv_path: str):
    """
    Parse the FER-2013 CSV into numpy arrays.

    CSV format (from Kaggle):
      emotion, pixels, Usage
      0, "70 80 82 ...", Training
    """
    import pandas as pd

    print(f"[Train] Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path)

    def row_to_image(pixels):
        arr = np.array(pixels.split(), dtype=np.float32)
        return arr.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0

    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    for _, row in df.iterrows():
        img   = row_to_image(row["pixels"])
        label = int(row["emotion"])
        usage = row.get("Usage", "Training")

        if usage == "Training":
            X_train.append(img); y_train.append(label)
        elif usage == "PublicTest":
            X_val.append(img);   y_val.append(label)
        else:
            X_test.append(img);  y_test.append(label)

    to_np = lambda lst: np.array(lst, dtype=np.float32)

    print(f"[Train] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return (
        to_np(X_train), keras.utils.to_categorical(y_train, NUM_CLASSES),
        to_np(X_val),   keras.utils.to_categorical(y_val,   NUM_CLASSES),
        to_np(X_test),  keras.utils.to_categorical(y_test,  NUM_CLASSES),
    )


# ══════════════════════════════════════════════════════════════
#  Model definition  (same arch as EmotionPredictor)
# ══════════════════════════════════════════════════════════════
def build_model() -> keras.Model:
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x   = layers.Flatten()(x)
    x   = layers.Dense(1024, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inp, out, name="EmotionCNN")
    model.compile(
        optimizer = optimizers.Adam(learning_rate=LR),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary()
    return model


# ══════════════════════════════════════════════════════════════
#  Data augmentation
# ══════════════════════════════════════════════════════════════
def make_augmentation_layer():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")


# ══════════════════════════════════════════════════════════════
#  Plot training history
# ══════════════════════════════════════════════════════════════
def plot_history(history, save_path: str = "logs/training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"],     label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[Train] History plot saved → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    if not os.path.exists(DATA_CSV):
        print(f"\n[Train] ❌  Dataset not found at {DATA_CSV}")
        print("[Train] Download FER-2013 from:")
        print("[Train]   https://www.kaggle.com/datasets/msambare/fer2013")
        print("[Train] Then place  fer2013.csv  inside the  data/  folder.\n")
        sys.exit(1)

    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(DATA_CSV)

    # ── Build model ───────────────────────────────────────────
    model = build_model()
    aug   = make_augmentation_layer()

    # ── Augmented training pipeline ───────────────────────────
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .batch(BATCH_SIZE)
        .map(lambda x, y: (aug(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Callbacks ─────────────────────────────────────────────
    cbs = [
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor          = "val_accuracy",
            save_best_only   = True,
            save_weights_only= True,
            verbose          = 1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 5,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        callbacks.EarlyStopping(
            monitor  = "val_accuracy",
            patience = 15,
            restore_best_weights = True,
            verbose  = 1,
        ),
        callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
    ]

    # ── Train ─────────────────────────────────────────────────
    print(f"\n[Train] Starting training for up to {EPOCHS} epochs …\n")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS,
        callbacks       = cbs,
    )

    # ── Evaluate on test set ──────────────────────────────────
    print("\n[Train] Evaluating on test set …")
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[Train] Test accuracy: {test_acc*100:.2f}%   Loss: {test_loss:.4f}")

    # ── Save & plot ───────────────────────────────────────────
    plot_history(history)
    print(f"\n[Train] ✅  Training complete!")
    print(f"[Train]    Best weights saved → {MODEL_PATH}")
    print("[Train]    Now run:  python emotion_detector.py\n")


if __name__ == "__main__":
    main()
