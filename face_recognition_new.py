import streamlit as st
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from PIL import Image
import time
import os
import numpy as np

st.title("📷 Face Recognition")
st.write("Record, test, and manage face recognition")


# ========================================
# Load Models
# ========================================
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet, device


mtcnn, resnet, device = load_models()

# ========================================
# Load Reference Embedding
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


reference_data = load_reference_embedding()

# ========================================
# Tabs
# ========================================
tab1, tab2, tab3 = st.tabs(["🎥 Test Recognition", "📷 Record Face", "ℹ️ Info"])

# ========================================
# TAB 1: TEST FACE RECOGNITION
# ========================================
with tab1:
    st.subheader("🎥 Test Face Recognition")

    if reference_data is None:
        st.error("❌ No face enrollment found.")
        st.info("Go to 'Enroll Biometrics' tab to enroll your face first.")
    else:
        st.write("Test your enrolled face against the camera")

        col1, col2 = st.columns([2, 1])

        with col1:
            video_placeholder = st.empty()
            status_placeholder = st.empty()

        with col2:
            st.subheader("Settings")
            similarity_placeholder = st.empty()
            threshold = st.slider("Similarity Threshold:", 0.5, 1.0, 0.65, 0.01, key="test_threshold")
            max_duration = st.slider("Max Duration (seconds):", 10, 60, 30, key="test_duration")

        if st.button("🎥 Start Test", key="face_test"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not cap.isOpened():
                st.error("❌ Cannot access camera. Please check your connection.")
            else:
                recognized = False
                frame_count = 0
                max_frames = max_duration * 30  # 30fps

                reference_emb = torch.tensor(reference_data["mean"], dtype=torch.float32).to(device)

                status_placeholder.info("🔍 Detecting face...")

                while frame_count < max_frames and not recognized:
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

                if not recognized:
                    st.warning(f"⏳ Face not recognized within {max_duration} seconds")

# ========================================
# TAB 2: RECORD FACE
# ========================================
with tab2:
    st.subheader("📷 Record/Update Face Enrollment")
    st.write("Record new face samples to update your enrollment")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Instructions:**")
        st.write("1. Sit in front of camera")
        st.write("2. Position your face in center")
        st.write("3. Look at camera with different angles")
        st.write("4. System will capture multiple samples automatically")

        num_samples = st.slider("Number of samples to record:", 5, 20, 10, key="record_samples")

    with col2:
        st.metric("Samples to Capture", num_samples)
        st.metric("Time Est.", f"~{num_samples * 2}s")

    if st.button("📷 Start Recording", key="face_record_btn"):
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

            status_text.info(f"📸 Recording: 0/{num_samples}")

            while captured < num_samples and frame_count < max_attempts:
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

                        progress = captured / num_samples
                        progress_bar.progress(progress)
                        status_text.success(f"✅ Captured: {captured}/{num_samples}")
                    else:
                        status_text.info(f"🔍 Detecting face... ({captured}/{num_samples})")

                except Exception as e:
                    status_text.error(f"Error: {str(e)}")

                time.sleep(0.2)

            cap.release()

            if captured >= num_samples:
                embeddings_array = np.array(embeddings)
                mean_embedding = embeddings_array.mean(axis=0)

                face_data = {
                    "mean": mean_embedding,
                    "all": embeddings_array,
                    "num_samples": captured
                }

                save_reference_embedding(face_data)
                st.success(f"✅ Face enrollment complete! ({captured} samples)")
                st.balloons()

                col1, col2, col3 = st.columns(3)
                col1.metric("Samples Recorded", captured)
                col2.metric("Status", "✅ Active")
                col3.metric("Embedding Dim", face_data["mean"].shape[0])
            else:
                st.error(f"❌ Only captured {captured}/{num_samples}. Please try again.")

    # Show current enrollment status
    st.divider()
    st.subheader("📊 Current Enrollment")

    if reference_data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples Enrolled", reference_data.get("num_samples", "Unknown"))
        col2.metric("Status", "✅ Active")
        col3.metric("Embedding Dim", reference_data["mean"].shape[0])

        st.divider()

        if st.button("🔄 Delete & Re-enroll", key="delete_face_btn"):
            os.remove(embedding_path)
            st.success("Face enrollment deleted. Click 'Start Recording' to re-enroll.")
            st.rerun()
    else:
        st.info("No face enrollment found. Click 'Start Recording' above to create one.")

# ========================================
# TAB 3: INFO
# ========================================
with tab3:
    st.subheader("ℹ️ How Face Recognition Works")

    st.markdown("""
    ### Process
    1. **Face Detection** - MTCNN detects face in video frame
    2. **Face Alignment** - Face is aligned to standard position
    3. **Embedding Extraction** - FaceNet extracts 512-dimensional face embedding
    4. **Similarity Comparison** - Cosine similarity between enrolled and test face
    5. **Threshold Check** - If similarity > threshold, face is recognized

    ### Parameters
    - **Similarity Threshold**: 0.5 (loose) to 1.0 (strict)
    - **Default**: 0.65 (recommended)
    - Higher = stricter matching

    ### Best Results
    - Good lighting conditions
    - Face centered in frame
    - Clear, direct view of face
    - No obstructions (glasses, masks may affect)
    - Record multiple angles during enrollment

    ### Workflow
    1. **First Time**: Go to "Record Face" tab → Click "Start Recording"
    2. **Test**: Go to "Test Recognition" tab → Click "Start Test"
    3. **Update**: Go to "Record Face" tab → Click "Start Recording" again

    ### Technical Details
    - **Model**: InceptionResnetV1 (pretrained on VGGFace2)
    - **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
    - **Embedding Size**: 512 dimensions
    - **Distance Metric**: Cosine Similarity
    """)