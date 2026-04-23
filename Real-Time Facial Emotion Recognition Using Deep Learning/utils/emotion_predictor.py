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
            result = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False, silent=True)
            emotions = result[0]["emotion"]
            label = max(emotions, key=emotions.get)
            confidence = emotions[label] / 100.0
            all_probs = [emotions.get(e.lower(), 0)/100.0 for e in EMOTIONS]
            return label.capitalize(), confidence, all_probs
        except:
            return "Neutral", 0.0, [0.0]*7
