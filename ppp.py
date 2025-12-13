import streamlit as st
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from PIL import Image
import time
import os
import numpy as np
import subprocess

# Page configuration
st.set_page_config(page_title="Face Recognition System", layout="wide")


# Load reference embedding
def load_reference_embedding():
    if not os.path.exists("reference_embedding.pkl"):
        return None
    with open("reference_embedding.pkl", "rb") as f:
        return pickle.load(f)


# Save reference embedding
def save_reference_embedding(data):
    with open("reference_embedding.pkl", "wb") as f:
        pickle.dump(data, f)


# Initialize models
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet, device


# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"

mtcnn, resnet, device = load_models()
reference_data = load_reference_embedding()

# ========================================
# SIDEBAR MENU
# ========================================
st.sidebar.title("🎭 Face Recognition System")
st.sidebar.markdown("---")

if not st.session_state.authenticated:
    st.session_state.current_page = "Login"
else:
    # Only update from sidebar if not already set by a button
    sidebar_choice = st.sidebar.radio(
        "Choose an action:",
        ["Dashboard", "Login", "Enroll Face", "Settings", "Launch Programs", "About"],
        index=["Dashboard", "Login", "Enroll Face", "Settings", "Launch Programs", "About"].index(
            st.session_state.current_page)
    )
    st.session_state.current_page = sidebar_choice

# ========================================
# LOGIN PAGE
# ========================================
if st.session_state.current_page == "Login":
    st.title("🔐 Face Login")
    st.write("Please sit in front of the camera for face recognition")

    if reference_data is None:
        st.error("❌ No reference face found. Please enroll a face first.")
        st.info("Go to 'Enroll Face' to create a new enrollment.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()

        with col2:
            st.subheader("Recognition Status")
            similarity_placeholder = st.empty()
            threshold = st.slider("Similarity Threshold:", 0.5, 1.0, 0.65, 0.01)

        start_button = st.button("🎥 Start Recognition", key="login_start")

        if start_button:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not cap.isOpened():
                st.error("❌ Cannot access camera. Please check your camera connection.")
            else:
                recognized = False
                frame_count = 0
                max_frames = 300  # 10 seconds at 30fps

                reference_emb = torch.tensor(reference_data["mean"], dtype=torch.float32).to(device)

                while frame_count < max_frames and not recognized:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera read failed")
                        break

                    frame_count += 1

                    # Convert BGR -> RGB
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

                    # Convert to PIL Image
                    pil_img = Image.fromarray(img_rgb)

                    # Detect face
                    try:
                        face = mtcnn(pil_img)

                        if face is not None:
                            # Compute embedding
                            with torch.no_grad():
                                emb = resnet(face.unsqueeze(0).to(device))[0]

                            # Cosine similarity
                            cos_sim = torch.nn.functional.cosine_similarity(emb, reference_emb, dim=0)
                            cos_sim_value = cos_sim.item()

                            similarity_placeholder.metric("Similarity Score", f"{cos_sim_value:.3f}")

                            if cos_sim_value > threshold:
                                recognized = True
                                status_placeholder.success(f"✅ Face Recognized! (Similarity: {cos_sim_value:.3f})")
                            else:
                                status_placeholder.warning(f"⏳ Face detected (Similarity: {cos_sim_value:.3f})")
                        else:
                            status_placeholder.info("🔍 Detecting face...")

                    except Exception as e:
                        status_placeholder.error(f"Error: {str(e)}")

                    time.sleep(0.1)

                cap.release()

                if recognized:
                    st.session_state.authenticated = True
                    st.success("🎉 Authentication successful! Redirecting...")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Face not recognized. Please try again.")

# ========================================
# ENROLL FACE PAGE
# ========================================
elif st.session_state.current_page == "Enroll Face":
    st.title("📝 Enroll New Face")
    st.write("Create a new face enrollment for future recognition")

    col1, col2 = st.columns([2, 1])

    with col1:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    with col2:
        st.subheader("Enrollment Info")
        num_samples = st.slider("Number of samples to capture:", 5, 20, 10)

    start_enroll = st.button("📷 Start Enrollment", key="enroll_start")

    if start_enroll:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Cannot access camera.")
        else:
            embeddings = []
            captured = 0
            frame_count = 0
            max_attempts = 500

            status_placeholder.info(f"📸 Capturing face: {captured}/{num_samples}")

            while captured < num_samples and frame_count < max_attempts:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

                pil_img = Image.fromarray(img_rgb)

                try:
                    face = mtcnn(pil_img)

                    if face is not None:
                        with torch.no_grad():
                            emb = resnet(face.unsqueeze(0).to(device))[0]
                        embeddings.append(emb.cpu().numpy())
                        captured += 1
                        status_placeholder.success(f"✅ Captured: {captured}/{num_samples}")
                    else:
                        status_placeholder.warning(f"⏳ Detecting face... ({captured}/{num_samples})")

                except Exception as e:
                    status_placeholder.error(f"Error: {str(e)}")

                time.sleep(0.2)

            cap.release()

            if captured >= num_samples:
                embeddings_array = np.array(embeddings)
                mean_embedding = embeddings_array.mean(axis=0)

                enrollment_data = {
                    "mean": mean_embedding,
                    "all": embeddings_array,
                    "num_samples": captured
                }

                save_reference_embedding(enrollment_data)
                st.success(f"✅ Enrollment complete! Captured {captured} face samples.")
                st.info("You can now use face login to authenticate.")
            else:
                st.error(f"❌ Only captured {captured}/{num_samples} samples. Please try again.")

# ========================================
# SETTINGS PAGE
# ========================================
elif st.session_state.current_page == "Settings":
    st.title("⚙️ Settings")

    if reference_data:
        st.subheader("📊 Current Enrollment Info")
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples Captured", reference_data.get("num_samples", "Unknown"))
        col2.metric("Embedding Dimension", reference_data["mean"].shape[0])
        col3.metric("Status", "✅ Active")

        st.divider()

        st.subheader("🔄 Re-enroll Face")
        st.write("Delete current enrollment and create a new one:")
        if st.button("Delete & Re-enroll", key="reenroll"):
            os.remove("reference_embedding.pkl")
            st.success("Current enrollment deleted. Go to 'Enroll Face' to create a new one.")
            st.rerun()
    else:
        st.warning("⚠️ No enrollment found.")
        st.info("Go to 'Enroll Face' to create an enrollment.")

    st.divider()
    st.subheader("ℹ️ System Info")
    st.write(f"**Device:** {device}")
    st.write(f"**CUDA Available:** {torch.cuda.is_available()}")

# ========================================
# DASHBOARD PAGE
# ========================================
elif st.session_state.current_page == "Dashboard":
    if st.session_state.authenticated:
        st.title("📊 Dashboard")
        st.write("Welcome to the Face Recognition System!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", "🟢 Authenticated")
        col2.metric("System", "Active")
        col3.metric("Device", device)

        st.divider()
        st.subheader("Quick Actions")
        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("🔐 Logout"):
                st.session_state.authenticated = False
                st.session_state.current_page = "Login"
                st.rerun()

        with action_col2:
            if st.button("⚙️ Settings"):
                st.session_state.current_page = "Settings"
                st.rerun()

        with action_col3:
            if st.button("ℹ️ About"):
                st.session_state.current_page = "About"
                st.rerun()
    else:
        st.warning("Please login first.")

# ========================================
# LAUNCH PROGRAMS PAGE
# ========================================
elif st.session_state.current_page == "Launch Programs":
    st.title("🚀 Launch Programs")
    st.write("Open and run other Streamlit applications")

    # Define available programs
    programs = {
        "ML DP Presentation": "page_0.py",
        "Problem Description": "page_1.py",
        "Research Data": "page_2.py",
        "Features Engineering": "page_3.py",
        "ML & DL Model Selection": "page_4.py",
        "Final Model Prediction": "page_5.py",
        "Behind The Scenes": "page_6.py",
        "Conclusions": "page_7.py",
    }

    st.subheader("📋 Available Applications")

    # Create columns for program cards
    cols = st.columns(2)
    col_idx = 0

    for prog_name, prog_file in programs.items():
        with cols[col_idx % 2]:
            with st.container(border=True):
                st.write(f"**{prog_name}**")
                st.caption(f"File: {prog_file}")

                if st.button(f"▶️ Launch", key=f"launch_{prog_name}"):
                    try:
                        # Check if file exists
                        if os.path.exists(prog_file):
                            # Launch the Streamlit app in a new process
                            subprocess.Popen(
                                ["streamlit", "run", prog_file],
                                cwd=os.getcwd()
                            )
                            st.success(f"✅ Launching {prog_name}...")
                            st.info(
                                f"📍 The app should open in your browser at a different port (usually :8502, :8503, etc.)")
                        else:
                            st.error(f"❌ File not found: {prog_file}")
                    except Exception as e:
                        st.error(f"❌ Error launching {prog_name}: {str(e)}")

        col_idx += 1

    st.divider()

    with st.expander("ℹ️ How to use"):
        st.markdown("""
        - Click **▶️ Launch** next to any program
        - A new browser tab/window will open with the application
        - Each app runs on a different port (8501, 8502, 8503, etc.)
        - You can run multiple programs at the same time

        **Common Ports:**
        - Face Recognition: http://localhost:8501
        - First launched program: http://localhost:8502
        - Second launched program: http://localhost:8503
        """)

# ========================================
# ABOUT PAGE
# ========================================
elif st.session_state.current_page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    ### Face Recognition System

    This application uses advanced deep learning models to authenticate users via facial recognition.

    **Technologies:**
    - PyTorch for deep learning
    - MTCNN for face detection
    - FaceNet (InceptionResnetV1) for face embedding
    - Streamlit for the web interface

    **Features:**
    - 🔐 Secure face-based authentication
    - 📝 Multi-sample enrollment
    - ⚙️ Configurable similarity threshold
    - 🎯 Real-time face detection
    """)

    if st.button("← Back to Dashboard"):
        st.session_state.current_page = "Dashboard"
        st.rerun()