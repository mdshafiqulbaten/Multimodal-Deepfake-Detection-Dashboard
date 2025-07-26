import streamlit as st
from models import face_detector, audio_detector, text_detector
from utils.fusion import fuse_predictions

st.set_page_config(page_title="Multimodal Deepfake Detection", layout="centered")
st.title("🎭 Multimodal Deepfake Detection Dashboard")
st.markdown("Upload a **video**, **audio**, and/or **transcript** to detect if it's a deepfake.")

# Upload Section
video_file = st.file_uploader("Upload Video File", type=["mp4", "mov"])
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
text_file = st.file_uploader("Upload Transcript File", type=["txt"])

# Detection Button
if st.button("🔍 Run Detection"):
    results = []

    if video_file:
        st.write("Analyzing video...")
        face_score = face_detector.detect_video(video_file)
        st.success(f"Face/Video Result: {face_score:.2f}")
        results.append(face_score)
    else:
        face_score = None

    if audio_file:
        st.write("Analyzing audio...")
        audio_score = audio_detector.detect_audio(audio_file)
        st.success(f"Audio Result: {audio_score:.2f}")
        results.append(audio_score)
    else:
        audio_score = None

    if text_file:
        st.write("Analyzing text...")
        transcript = text_file.read().decode("utf-8")
        text_score = text_detector.detect_text(transcript)
        st.success(f"Text Result: {text_score:.2f}")
        results.append(text_score)
    else:
        text_score = None

    if results:
        final_score, verdict = fuse_predictions(results)
        st.header(f"🧠 Final Verdict: {verdict} ({final_score:.2f})")
    else:
        st.warning("Please upload at least one modality.")
