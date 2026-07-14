import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

st.title("😊 Real-Time Facial Emotion Recognition")
st.write("Upload an image to detect the emotion.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    try:
        result = DeepFace.analyze(
            img,
            actions=["emotion"],
            enforce_detection=False
        )

        emotions = result[0]["emotion"]

        emotion = max(emotions, key=emotions.get)

        st.success(f"Detected Emotion: **{emotion.capitalize()}**")

        st.subheader("Confidence Scores")

        st.bar_chart(emotions)

    except Exception as e:
        st.error(str(e))