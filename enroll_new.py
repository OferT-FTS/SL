import streamlit as st
import cv2
import torch
import numpy as np
import pickle
import os
import time
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import librosa

# ========================================
# Page Setup
# ========================================
st.title("📝 Enroll Biometrics")
st.write("Enroll or re-enroll your face and voice for authentication")


# ========================================
# Initialize Models
# ========================================
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet, device


mtcnn, resnet, device = load_models()

# ========================================
# File Paths
# ========================================
face_embedding_path = "reference_embedding.pkl"
voice_embedding_path = "voice_embedding.pkl"


# ========================================
# Helper Functions
# ========================================

def load_embedding(path):
    """Load embedding from file"""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_embedding(data, path):
    """Save embedding to file"""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def extract_voice_features(audio_data, sr=16000):
    """Extract MFCC features from audio"""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def record_audio(duration=5, sr=16000):
    """Record audio from microphone"""
    try:
        import sounddevice as sd
        st.write(f"🎤 Recording for {duration} seconds... Please speak clearly")
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
        sd.wait()
        return audio_data.flatten(), sr
    except Exception as e:
        st.error(f"❌ Error recording audio: {str(e)}")
        return None, None


# ========================================
# Tabs for Face and Voice Enrollment
# ========================================

tab1, tab2 = st.tabs(["👤 Face Enrollment", "🎤 Voice Enrollment"])

# ========================================
# TAB 1: FACE ENROLLMENT
# ========================================

with tab1:
    st.subheader("📷 Face Enrollment")
    st.write("Record face samples for authentication")

    existing_face = load_embedding(face_embedding_path)

    if existing_face:
        st.success("✅ Face already enrolled!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", existing_face.get("num_samples", "?"))
        col2.metric("Status", "Active")
        col3.metric("Embedding Dim", existing_face["mean"].shape[0])
        st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Instructions:**")
        st.write("1. Sit in front of camera")
        st.write("2. Position your face in center")
        st.write("3. System will capture multiple angles")
        st.write("4. Click button to start")

        num_face_samples = st.slider("Number of face samples:", 5, 20, 10, key="face_samples")

    with col2:
        st.metric("Samples to Capture", num_face_samples)
        st.metric("Time Est.", f"~{num_face_samples * 2}s")

    if st.button("📷 Start Face Enrollment", key="face_enroll_btn"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Cannot access camera")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            video_frame = st.empty()

            embeddings = []
            captured = 0
            frame_count = 0
            max_attempts = 500

            while captured < num_face_samples and frame_count < max_attempts:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame.image(img_rgb, channels="RGB", use_column_width=True)

                pil_img = Image.fromarray(img_rgb)

                try:
                    face = mtcnn(pil_img)

                    if face is not None:
                        with torch.no_grad():
                            emb = resnet(face.unsqueeze(0).to(device))[0]
                        embeddings.append(emb.cpu().numpy())
                        captured += 1

                        progress = captured / num_face_samples
                        progress_bar.progress(progress)
                        status_text.success(f"✅ Captured: {captured}/{num_face_samples}")
                    else:
                        status_text.info(f"🔍 Detecting face... ({captured}/{num_face_samples})")

                except Exception as e:
                    status_text.error(f"Error: {str(e)}")

                time.sleep(0.2)

            cap.release()

            if captured >= num_face_samples:
                embeddings_array = np.array(embeddings)
                mean_embedding = embeddings_array.mean(axis=0)

                face_data = {
                    "mean": mean_embedding,
                    "all": embeddings_array,
                    "num_samples": captured
                }

                save_embedding(face_data, face_embedding_path)
                st.success(f"✅ Face enrollment complete! ({captured} samples)")
                st.balloons()
            else:
                st.error(f"❌ Only captured {captured}/{num_face_samples}. Please try again.")

    if existing_face:
        st.divider()
        if st.button("🔄 Delete & Re-enroll Face", key="delete_face"):
            os.remove(face_embedding_path)
            st.success("Face enrollment deleted. Refresh to re-enroll.")
            st.rerun()

# ========================================
# TAB 2: VOICE ENROLLMENT
# ========================================

with tab2:
    st.subheader("🎤 Voice Enrollment")
    st.write("Record voice samples for authentication")

    existing_voice = load_embedding(voice_embedding_path)

    if existing_voice:
        st.success("✅ Voice already enrolled!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", existing_voice.get("num_samples", "?"))
        col2.metric("Status", "Active")
        col3.metric("Threshold", f"{existing_voice.get('threshold', 0.7):.2f}")
        st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Instructions:**")
        st.write("1. Find a quiet environment")
        st.write("2. Speak clearly and naturally")
        st.write("3. Record multiple samples")
        st.write("4. Click button to start")

        num_voice_samples = st.slider("Number of voice samples:", 3, 10, 5, key="voice_samples")
        voice_duration = st.slider("Duration per sample (seconds):", 3, 10, 5, key="voice_duration")
        voice_threshold = st.slider("Verification threshold:", 0.5, 0.95, 0.7, 0.05, key="voice_threshold")

    with col2:
        st.metric("Samples to Record", num_voice_samples)
        st.metric("Duration Each", f"{voice_duration}s")
        st.metric("Total Time", f"~{num_voice_samples * voice_duration}s")

    if st.button("🎤 Start Voice Enrollment", key="voice_enroll_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        audio_container = st.container()

        embeddings = []

        for i in range(num_voice_samples):
            status_text.info(f"📸 Recording sample {i + 1}/{num_voice_samples}...")

            audio, sr = record_audio(duration=voice_duration)

            if audio is None:
                st.error("❌ Failed to record audio")
                st.stop()

            try:
                features = extract_voice_features(audio, sr=sr)
                embeddings.append(features)

                with audio_container:
                    st.write(f"**Sample {i + 1}:** Recorded")
                    st.audio(audio, sample_rate=sr)

                progress = (i + 1) / num_voice_samples
                progress_bar.progress(progress)
                status_text.success(f"✅ Sample {i + 1}/{num_voice_samples}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        embeddings_array = np.array(embeddings)
        mean_embedding = embeddings_array.mean(axis=0)

        voice_data = {
            "mean": mean_embedding.tolist(),
            "all": embeddings_array.tolist(),
            "num_samples": num_voice_samples,
            "threshold": voice_threshold,
            "duration": voice_duration
        }

        save_embedding(voice_data, voice_embedding_path)
        st.success("🎉 Voice enrollment complete!")
        st.balloons()

    if existing_voice:
        st.divider()
        if st.button("🔄 Delete & Re-enroll Voice", key="delete_voice"):
            os.remove(voice_embedding_path)
            st.success("Voice enrollment deleted. Refresh to re-enroll.")
            st.rerun()

# ========================================
# Info Section
# ========================================
with st.expander("ℹ️ How It Works"):
    st.markdown("""
    ### Face Recognition
    - Uses MTCNN for face detection
    - FaceNet embeddings for identification
    - Cosine similarity for matching

    ### Voice Recognition
    - MFCC (Mel-Frequency Cepstral Coefficients) extraction
    - Multiple sample averaging for accuracy
    - Threshold-based authentication

    ### Best Practices
    - Enroll in good lighting for faces
    - Enroll in quiet environment for voice
    - Record multiple angles/variations
    - Use consistent speaking voice
    """)