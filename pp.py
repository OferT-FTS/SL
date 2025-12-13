import streamlit as st
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from PIL import Image
import time
import os
import numpy as np

# Page configuration
st.set_page_config(page_title="Face Recognition System", layout="wide")

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
                st.success("🎉 Authentication successful! Loading presentation...")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Face not recognized. Please try again.")

# ========================================
# After authentication - Multi-page navigation
# ========================================
if st.session_state.authenticated:
    st.logo("combined_logo.png")

    # Define pages
    pages = [
        st.Page("page_0.py", title="ML DP Presentation", icon="📊"),
        st.Page("page_1.py", title="Problem Description", icon="📋"),
        st.Page("page_2.py", title="Research Data", icon="📈"),
        st.Page("page_3.py", title="Features Engineering", icon="🔧"),
        st.Page("page_4.py", title="ML & DL Models", icon="🤖"),
        st.Page("page_5.py", title="Model Prediction", icon="🎯"),
        st.Page("page_6.py", title="Behind The Scenes", icon="🔬"),
        st.Page("page_7.py", title="Conclusions", icon="✨"),
    ]

    # Add settings and enrollment pages
    settings_page = st.Page("ppp.py", title="Face Recognition App", icon="🎭")
    # enroll_page = st.Page("enroll.py", title="Enroll Face", icon="📝")

    # Navigation with all pages (divider not needed in st.navigation)
    # pg = st.navigation(pages + [enroll_page, settings_page])
    pg = st.navigation(pages + [settings_page])
    pg.run()