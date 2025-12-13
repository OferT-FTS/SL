import streamlit as st
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from PIL import Image
import time
import os
import numpy as np
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine
import pyttsx3
import threading

# Page configuration
st.set_page_config(page_title="Face Recognition System", layout="wide")


# ========================================
# Text-to-Speech Function
# ========================================
def speak(text):
    """Convert text to speech"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")


# ========================================
# Load reference embedding
# ========================================
embedding_path = "reference_embedding.pkl"


def load_reference_embedding():
    if not os.path.exists(embedding_path):
        return None
    with open(embedding_path, "rb") as f:
        return pickle.load(f)


def save_reference_embedding(data):
    with open(embedding_path, "wb") as f:
        pickle.dump(data, f)


# ========================================
# Voice Recognition Functions
# ========================================
voice_embedding_path = "voice_embedding.pkl"


def load_voice_embedding():
    if not os.path.exists(voice_embedding_path):
        return None
    with open(voice_embedding_path, "rb") as f:
        return pickle.load(f)


def save_voice_embedding(data):
    with open(voice_embedding_path, "wb") as f:
        pickle.dump(data, f)


def extract_voice_features(audio_data, sr=16000):
    """Extract MFCC features from audio"""
    # Compute MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    # Return mean of each coefficient as feature vector
    return np.mean(mfcc, axis=1)


def record_audio(duration=5, sr=16000):
    """Record audio from microphone"""
    try:
        import sounddevice as sd
        st.write(f"🎤 Recording for {duration} seconds... Please speak clearly")
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        return audio_data.flatten(), sr
    except Exception as e:
        st.error(f"❌ Error recording audio: {str(e)}")
        return None, None


def compare_voice(test_audio, reference_embedding, sr=16000):
    """Compare test audio with reference embedding"""
    test_features = extract_voice_features(test_audio, sr=sr)
    # Calculate similarity (1 - cosine distance)
    distance = cosine(test_features, reference_embedding)
    similarity = 1 - distance
    return similarity


# def check_voice_activity(audio_data, sr=16000, threshold=0.02):
#     """Check if there is actual voice activity in the audio"""
#     # Remove very low amplitude samples
#     audio_data = audio_data.astype(np.float32)
#
#     # Calculate RMS (Root Mean Square) energy
#     rms = np.sqrt(np.mean(audio_data ** 2))
#
#     st.write(f"**Debug - RMS Energy:** {rms:.4f} (threshold: {threshold})")
#
#     # If RMS is below threshold, it's likely silence
#     if rms < threshold:
#         st.warning(f"RMS too low ({rms:.4f} < {threshold})")
#         return False
#
#     # Check audio peaks
#     peak = np.max(np.abs(audio_data))
#     st.write(f"**Debug - Peak Amplitude:** {peak:.4f}")
#
#     if peak < 0.05:
#         st.warning(f"Peak too low ({peak:.4f})")
#         return False
#
#     # Calculate zero crossing rate (speech has higher ZCR than silence)
#     zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
#     st.write(f"**Debug - Zero Crossing Rate:** {zcr:.4f}")
#
#     if zcr < 0.05:
#         st.warning("Too few zero crossings - likely silence")
#         return False
#
#     st.success(f"✅ Voice activity detected! (RMS: {rms:.4f}, Peak: {peak:.4f}, ZCR: {zcr:.4f})")
#     return True
def check_voice_activity(audio_data, sr=16000):
    """
    Strong Voice Activity Detection (VAD)
    Rejects:
        - Silence
        - Background noise
        - Loud pops / peaks
        - Screaming / non-speech
    Accepts:
        - Natural human speech only
    """

    audio_data = audio_data.astype(np.float32)

    # 1. RMS energy – basic intensity of speech
    rms = np.sqrt(np.mean(audio_data ** 2))

    # 2. Peak amplitude – reject loud nonspeech spikes
    peak = np.max(np.abs(audio_data))

    # 3. Zero Crossing Rate – speech has moderate ZCR (0.1–0.25)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

    # 4. Spectral centroid – speech: 200–3000 Hz
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))

    st.write(f"**RMS:** {rms:.4f}")
    st.write(f"**Peak:** {peak:.4f}")
    st.write(f"**ZCR:** {zcr:.4f}")
    st.write(f"**Spectral Centroid:** {spec_centroid:.2f} Hz")

    # ========================
    #      DECISION RULES
    # ========================

    # Silence: very low RMS
    if rms < 0.02:
        st.warning("⛔ RMS too low → silence detected")
        return False

    # Peaks too high → screaming/noise
    if peak > 0.98:
        st.warning("⛔ Peak too high → loud noise or scream detected")
        return False

    # Background noise usually has low ZCR + low centroid
    if zcr < 0.05:
        st.warning("⛔ ZCR too low → no speech")
        return False

    # Screaming / non-speech has very high centroid
    if spec_centroid > 4000:
        st.warning("⛔ Sound too sharp → not speech")
        return False

    # Must be within typical speech frequency range
    if not (200 < spec_centroid < 3000):
        st.warning("⛔ Not typical speech frequency profile")
        return False

    st.success("✅ Clear human speech detected!")
    return True


# ========================================
# Setup models (cached)
# ========================================
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet, device


device_used = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn, resnet, device = load_models()
reference_data = load_reference_embedding()

# ========================================
# Streamlit session state
# ========================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'face_recognized' not in st.session_state:
    st.session_state.face_recognized = False
if 'voice_verified' not in st.session_state:
    st.session_state.voice_verified = False
if 'greeting_played' not in st.session_state:
    st.session_state.greeting_played = False
if 'voice_greeting_played' not in st.session_state:
    st.session_state.voice_greeting_played = False

# ========================================
# Authentication / Login Screen
# ========================================
if not st.session_state.authenticated:
    st.title("🔐 Face Recognition Login")
    st.write("Please sit in front of the camera for face recognition")

    if reference_data is None:
        st.error("❌ No reference face found. Please enroll first.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col2:
        st.subheader("Recognition Info")
        similarity_placeholder = st.empty()
        threshold = st.slider("Similarity Threshold:", 0.5, 1.0, 0.65, 0.01)

    start_button = st.button("🎥 Start Recognition", key="auth_start")

    if start_button:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check your connection.")
        else:
            authenticated = False
            frame_count = 0
            max_frames = 300  # 10 seconds at 30fps

            reference_emb = torch.tensor(reference_data["mean"], dtype=torch.float32).to(device)

            status_placeholder.info("🔍 Detecting face...")

            while frame_count < max_frames and not authenticated:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera read failed")
                    break

                frame_count += 1

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

                pil_img = Image.fromarray(img_rgb)

                try:
                    face = mtcnn(pil_img)

                    if face is not None:
                        with torch.no_grad():
                            emb = resnet(face.unsqueeze(0).to(device))[0]

                        cos_sim = torch.nn.functional.cosine_similarity(emb, reference_emb, dim=0)
                        cos_sim_value = cos_sim.item()

                        similarity_placeholder.metric("Similarity Score", f"{cos_sim_value:.3f}")

                        if cos_sim_value > threshold:
                            authenticated = True
                            st.session_state.authenticated = True
                            status_placeholder.success(f"✅ Face Recognized! ({cos_sim_value:.3f})")
                            break
                        else:
                            status_placeholder.warning(f"⏳ Face detected ({cos_sim_value:.3f})")
                    else:
                        status_placeholder.info("🔍 Detecting face...")

                except Exception as e:
                    status_placeholder.error(f"Error: {str(e)}")

                time.sleep(0.1)

            cap.release()

            if authenticated:
                st.success("🎉 Face Recognized! Moving to voice verification...")
                threading.Thread(target=lambda: speak("Face recognized. Now please verify with voice recognition"),
                                 daemon=True).start()
                time.sleep(2)
                st.session_state.face_recognized = True
                st.rerun()
            else:
                st.error("❌ Face not recognized. Please try again.")

# ========================================
# Voice Verification (after face recognition)
# ========================================
if st.session_state.face_recognized and not st.session_state.voice_verified:
    st.title("🎤 Voice Verification")
    st.write("Please verify your identity with voice recognition")

    voice_data = load_voice_embedding()

    if voice_data is None:
        st.error("❌ No voice profile found. Please enroll your voice first.")
        st.info("Go to Settings to enroll your voice.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎙️ Record Your Voice")
        st.write("Please say a passphrase or speak naturally for 5 seconds")

        record_duration = st.slider("Recording duration (seconds):", 3, 10, 5)

        if st.button("🎤 Start Voice Recording", key="voice_record"):
            audio, sr = record_audio(duration=record_duration)

            if audio is not None:
                st.success("✅ Recording complete!")
                st.audio(audio, sample_rate=sr)

                # Compare voice
                reference_voice_emb = np.array(voice_data["mean"])
                similarity = compare_voice(audio, reference_voice_emb, sr=sr)

                st.write(f"**Voice Similarity:** {similarity:.3f}")

                threshold = voice_data.get("threshold", 0.7)

                if similarity > threshold:
                    st.success(f"✅ Voice Verified! (Similarity: {similarity:.3f})")
                    st.session_state.voice_verified = True
                    st.session_state.authenticated = True

                    # Welcome message
                    threading.Thread(target=lambda: speak("Welcome! All the best with your presentation. Good luck!"),
                                     daemon=True).start()
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Voice not recognized. (Similarity: {similarity:.3f}, Required: {threshold:.3f})")
                    st.info("Please try again")

    with col2:
        st.subheader("ℹ️ Tips")
        st.markdown("""
        - Speak clearly and naturally
        - Minimize background noise
        - Use same tone as enrollment
        - Don't rush your words
        """)

# ========================================
# After complete authentication
# ========================================
if st.session_state.authenticated and st.session_state.voice_verified:
    st.logo("combined_logo.png")

    # Define pages
    pages = [
        st.Page("page_0.py", title=f"ML DP Presentation", icon="🔹"),
        st.Page("page_1.py", title="Problem Description", icon="🔹"),
        st.Page("page_2.py", title="Research Data", icon="🔹"),
        st.Page("page_3.py", title="Features Engineering", icon="🔹"),
        st.Page("page_4.py", title="ML & DL Models", icon="🔹"),
        st.Page("page_5.py", title="Model Prediction", icon="🔹"),
        st.Page("page_6.py", title="Behind The Scenes", icon="🔹"),
        st.Page("page_7.py", title="Conclusions", icon="🔹"),
        st.Page("page_8.py", title="References", icon="🔹"),
    ]

    # Add settings and enrollment pages
    settings_page = st.Page("settings.py", title="Settings", icon="⚙️")
    enroll_page = st.Page("enroll_new.py", title="Enroll Voice", icon="🎤")
    face_page = st.Page("face_recognition_new.py", title="Face Recognition", icon="🎭")

    # Navigation with all pages
    pg = st.navigation(pages + [face_page, enroll_page, settings_page])

    # Add logout button in sidebar
    # st.sidebar.divider()

    if st.sidebar.button("🔐 Logout", key="logout_btn", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.face_recognized = False
        st.session_state.voice_verified = False
        st.rerun()
    # st.sidebar.divider()
    pg.run()