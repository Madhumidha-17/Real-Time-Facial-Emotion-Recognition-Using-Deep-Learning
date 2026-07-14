from deepface import DeepFace
import cv2

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

class EmotionPredictor:
    def __init__(self):
        self.EMOTIONS = EMOTIONS
        print("[EmotionPredictor] Using DeepFace model")

    def predict(self, face_roi):
        try:
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(
                rgb,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True
            )
            emotions = result[0]["emotion"]

            # DeepFace returns lowercase keys — find the highest one
            label = max(emotions, key=emotions.get)  # e.g. "angry"
            confidence = emotions[label] / 100.0

            # Map to our capitalized EMOTIONS list
            label_cap = label.capitalize()  # "Angry"

            # Build all_probs in EMOTIONS order
            all_probs = [emotions.get(e.lower(), 0.0) / 100.0 for e in EMOTIONS]

            # Debug: print all scores to terminal
            scores = {e: f"{emotions.get(e.lower(),0):.1f}%" for e in EMOTIONS}
            print(f"[Emotion Scores] {scores}")

            return label_cap, confidence, all_probs

        except Exception as ex:
            print(f"[EmotionPredictor ERROR] {ex}")
            return "Neutral", 0.0, [0.0] * 7