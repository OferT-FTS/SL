import streamlit as st
import os
import pickle

st.title("⚙️ Settings")
st.write("Manage your authentication and system settings")

# ========================================
# Face Recognition Settings
# ========================================
st.subheader("👤 Face Recognition")

face_embedding_path = "reference_embedding.pkl"

if os.path.exists(face_embedding_path):
    with open(face_embedding_path, "rb") as f:
        face_data = pickle.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Face Samples", face_data.get("num_samples", "Unknown"))
    col2.metric("Status", "✅ Enrolled")
    col3.metric("Embedding Dim", face_data["mean"].shape[0])

    st.divider()

    if st.button("🔄 Re-enroll Face", key="reenroll_face"):
        os.remove(face_embedding_path)
        st.success("✅ Face enrollment deleted. Please re-authenticate to re-enroll.")
        st.rerun()
else:
    st.warning("⚠️ No face enrollment found")
    st.info("Please complete face enrollment during login")

# ========================================
# Voice Recognition Settings
# ========================================
st.divider()
st.subheader("🎤 Voice Recognition")

voice_embedding_path = "voice_embedding.pkl"

if os.path.exists(voice_embedding_path):
    with open(voice_embedding_path, "rb") as f:
        voice_data = pickle.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Voice Samples", voice_data.get("num_samples", "Unknown"))
    col2.metric("Status", "✅ Enrolled")
    col3.metric("Threshold", f"{voice_data.get('threshold', 0.7):.2f}")

    st.divider()

    if st.button("🔄 Re-enroll Voice", key="reenroll_voice"):
        os.remove(voice_embedding_path)
        st.success("✅ Voice enrollment deleted. Please go to 'Enroll Voice' to re-enroll.")
        st.rerun()
else:
    st.warning("⚠️ No voice enrollment found")
    st.info("Go to 'Enroll Voice' to create voice enrollment")

# ========================================
# System Information
# ========================================
st.divider()
st.subheader("ℹ️ System Information")

try:
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    col1, col2 = st.columns(2)
    col1.metric("Device", device.upper())
    col2.metric("CUDA Available", "✅ Yes" if torch.cuda.is_available() else "❌ No")
except:
    st.info("PyTorch not available")

# ========================================
# About
# ========================================
st.divider()
st.subheader("📋 About")

st.markdown("""
### Face & Voice Recognition System

**Authentication Methods:**
- 👤 **Face Recognition** - MTCNN + FaceNet embeddings
- 🎤 **Voice Recognition** - MFCC features + Cosine similarity

**Technologies:**
- PyTorch for deep learning
- OpenCV for computer vision
- Librosa for audio processing
- Streamlit for web interface

**Security:**
- Multi-factor authentication (Face + Voice)
- Local storage of embeddings
- Real-time verification

---
*Version 1.0 | Made with ❤️*
""")