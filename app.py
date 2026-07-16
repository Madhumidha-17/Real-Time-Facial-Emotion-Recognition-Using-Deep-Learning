import streamlit as st
import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Real-Time Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide"
)

# ----------------------------------------------------
# CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main-title{
text-align:center;
font-size:50px;
font-weight:bold;
background:linear-gradient(90deg,#2563EB,#9333EA);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
text-align:center;
font-size:19px;
color:#6B7280;
margin-bottom:25px;
}

.feature-box{
background:#F8FAFC;
padding:18px;
border-radius:12px;
border:1px solid #E5E7EB;
text-align:center;
}

.footer{
text-align:center;
color:gray;
font-size:15px;
margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Header
# ----------------------------------------------------

st.markdown(
"<h1 class='main-title'>🎭 Real-Time Facial Emotion Recognition</h1>",
unsafe_allow_html=True
)

st.markdown(
"<p class='subtitle'>AI-powered Facial Emotion Detection using <b>DeepFace</b> and <b>Streamlit</b>.</p>",
unsafe_allow_html=True
)

# ----------------------------------------------------
# Feature Cards
# ----------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
### 📤 Upload Image

Upload a face image and instantly predict the person's emotion.
""")

with c2:
    st.success("""
### 🎥 Live Webcam

Detect emotions continuously using your webcam in real time.
""")

with c3:
    st.warning("""
### 🤖 AI Prediction

Powered by DeepFace and Deep Learning for accurate emotion recognition.
""")

st.divider()

# ----------------------------------------------------
# Detection Mode
# ----------------------------------------------------

mode = st.radio(
    "📌 Select Detection Mode",
    ["📤 Upload Image", "🎥 Live Webcam"],
    horizontal=True
)

# ====================================================
# IMAGE UPLOAD
# ====================================================

if mode == "📤 Upload Image":

    st.markdown("### 📤 Upload a Face Image")

    uploaded_file = st.file_uploader(
        "Choose JPG / JPEG / PNG Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        img = np.array(image)

        left,right = st.columns([1,1])

        with left:
            st.image(
                img,
                caption="🖼 Uploaded Image",
                use_container_width=True
            )

        try:

            result = DeepFace.analyze(
                img,
                actions=["emotion"],
                enforce_detection=False
            )

            emotions = result[0]["emotion"]

            emotion = max(emotions,key=emotions.get)

            confidence = emotions[emotion]

            emoji={
                "happy":"😊",
                "sad":"😢",
                "angry":"😠",
                "fear":"😨",
                "surprise":"😲",
                "neutral":"😐",
                "disgust":"🤢"
            }

            with right:

                st.subheader("🎯 Prediction")

                st.success(
                    f"{emoji.get(emotion.lower(),'🙂')} **{emotion.capitalize()}**"
                )

                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

                st.progress(confidence/100)

                st.subheader("📊 Emotion Confidence")

                st.bar_chart(emotions)

        except Exception as e:

            st.error(e)

# ====================================================
# WEBCAM
# ====================================================

else:

    st.markdown("## 🎥 Live Webcam Emotion Detection")

    st.info("""
✅ Click **START** below.

😊 Keep your face visible.

💡 Good lighting improves prediction accuracy.
""")

    class EmotionDetector(VideoTransformerBase):

        def transform(self, frame):

            img = frame.to_ndarray(format="bgr24")

            try:

                result = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    enforce_detection=False
                )

                emotion=result[0]["dominant_emotion"]

                cv2.putText(
                    img,
                    f"Emotion : {emotion.capitalize()}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2
                )

            except Exception:
                pass

            return img

    webrtc_streamer(
        key="emotion",
        video_processor_factory=EmotionDetector,
        async_processing=True
    )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.divider()

st.markdown("""
<div class="footer">

💙 <b>Developed by Madhumidha</b>

🐍 Python &nbsp; | &nbsp; 🎭 DeepFace &nbsp; | &nbsp; 🚀 Streamlit

<i>Real-Time Facial Emotion Recognition Using Deep Learning</i>

</div>
""", unsafe_allow_html=True)