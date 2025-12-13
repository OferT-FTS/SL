import streamlit as st
import numpy as np
import pickle
import os
import librosa

# ========================================
# Voice Enrollment Page
# ========================================

st.title("📝 Voice Enrollment")
st.write("Record and enroll your voice for future authentication")

voice_embedding_path = "voice_embedding.pkl"


# ========================================
# Helper Functions
# ========================================

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
        sd.wait()  # Wait until recording is finished
        return audio_data.flatten(), sr
    except Exception as e:
        st.error(f"❌ Error recording audio: {str(e)}")
        return None, None


def load_voice_embedding():
    """Load existing voice embedding"""
    if not os.path.exists(voice_embedding_path):
        return None
    with open(voice_embedding_path, "rb") as f:
        return pickle.load(f)


def save_voice_embedding(data):
    """Save voice embedding"""
    with open(voice_embedding_path, "wb") as f:
        pickle.dump(data, f)


# ========================================
# Check existing enrollment
# ========================================

existing_voice = load_voice_embedding()

if existing_voice:
    st.success("✅ Voice already enrolled!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples Recorded", existing_voice.get("num_samples", "Unknown"))
    col2.metric("Threshold", f"{existing_voice.get('threshold', 0.7):.2f}")
    col3.metric("Status", "Active")

    st.divider()

# ========================================
# Voice Enrollment Interface
# ========================================

st.subheader("🎙️ Record Your Voice")

col1, col2 = st.columns([2, 1])

with col1:
    st.write("Please record samples of your voice. You'll need to record multiple samples for better accuracy.")

    # Settings
    num_samples = st.slider("Number of samples to record:", 3, 10, 5)
    record_duration = st.slider("Duration per sample (seconds):", 3, 10, 5)
    threshold = st.slider("Verification threshold:", 0.5, 0.95, 0.7, 0.05)

    st.info("""
    **Tips for best results:**
    - Find a quiet environment
    - Speak naturally and clearly
    - Vary your speaking slightly between samples
    - Use the same voice for all samples
    """)

with col2:
    st.subheader("📊 Recording Info")
    st.metric("Samples to Record", num_samples)
    st.metric("Duration Each", f"{record_duration}s")
    st.metric("Total Time", f"~{num_samples * record_duration}s")

st.divider()

# ========================================
# Start Enrollment Button
# ========================================

if st.button("🎤 Start Voice Enrollment", key="start_enroll"):
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    audio_container = st.container()

    embeddings = []

    for i in range(num_samples):
        status_placeholder.info(f"📸 Recording sample {i + 1}/{num_samples}...")

        # Record audio
        audio, sr = record_audio(duration=record_duration)

        if audio is None:
            st.error("❌ Failed to record audio. Please check your microphone.")
            st.stop()

        # Extract features
        try:
            features = extract_voice_features(audio, sr=sr)
            embeddings.append(features)

            # Display audio
            with audio_container:
                st.write(f"**Sample {i + 1}:** Recorded")
                st.audio(audio, sample_rate=sr)

            # Update progress
            progress = (i + 1) / num_samples
            progress_bar.progress(progress)
            status_placeholder.success(f"✅ Sample {i + 1}/{num_samples} recorded!")

        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
            st.stop()

    # Calculate mean embedding
    embeddings_array = np.array(embeddings)
    mean_embedding = embeddings_array.mean(axis=0)

    # Save enrollment data
    enrollment_data = {
        "mean": mean_embedding.tolist(),
        "all": embeddings_array.tolist(),
        "num_samples": num_samples,
        "threshold": threshold,
        "duration": record_duration
    }

    save_voice_embedding(enrollment_data)

    st.success("🎉 Voice enrollment complete!")
    st.balloons()

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples Recorded", num_samples)
    col2.metric("Threshold Set", f"{threshold:.2f}")
    col3.metric("Status", "✅ Ready")

    st.info("Your voice is now enrolled. You can use it for authentication on the login page.")

# ========================================
# Re-enrollment Option
# ========================================

if existing_voice:
    st.divider()
    st.subheader("🔄 Re-enroll Voice")

    if st.button("Delete & Re-enroll", key="delete_voice"):
        os.remove(voice_embedding_path)
        st.success("Voice enrollment deleted. Refresh the page to start a new enrollment.")
        st.rerun()

# ========================================
# Additional Info
# ========================================

with st.expander("ℹ️ How Voice Recognition Works"):
    st.markdown("""
    ### Voice Recognition Process

    **Enrollment Phase:**
    1. Record multiple voice samples (3-10)
    2. Extract voice features (MFCC - Mel-Frequency Cepstral Coefficients)
    3. Calculate average voice profile
    4. Store profile for future authentication

    **Authentication Phase:**
    1. Record a test voice sample
    2. Extract features from test sample
    3. Compare with stored profile (cosine similarity)
    4. If similarity > threshold, authentication succeeds

    **Why Multiple Samples?**
    - Better accuracy
    - Handles natural voice variations
    - More robust recognition

    **Voice Feature (MFCC):**
    - Mimics how human ear perceives sound
    - Captures voice characteristics
    - Resistant to background noise (relatively)
    """)