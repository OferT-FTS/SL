import streamlit as st
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from PIL import Image
import time
import os

# -----------------------------
# Load reference embedding
# -----------------------------
embedding_path = "reference_embedding.pkl"
if not os.path.exists(embedding_path):
    st.warning("No reference embedding found. Please enroll first.")
    st.stop()

with open(embedding_path, "rb") as f:
    reference_data = pickle.load(f)

if not isinstance(reference_data, dict) or "mean" not in reference_data:
    st.error("reference_embedding.pkl must contain a dict with key 'mean'")
    st.stop()

reference_emb = torch.tensor(reference_data["mean"], dtype=torch.float32)

# -----------------------------
# Setup models
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
reference_emb = reference_emb.to(device)

# -----------------------------
# Streamlit session state
# -----------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# -----------------------------
# Login / authentication screen
# -----------------------------
if not st.session_state.authenticated:
    login_placeholder = st.empty()  # placeholder for the whole login section

    with login_placeholder.container():
        st.title("Face Login - Please sit in front of the camera")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        video_placeholder = st.empty()

        authenticated = False
        while not authenticated:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera read failed")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(img_rgb, channels="RGB")  # show live frame

            pil_img = Image.fromarray(img_rgb)
            face = mtcnn(pil_img)

            if face is not None:
                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device))[0]

                cos_sim = torch.nn.functional.cosine_similarity(emb, reference_emb, dim=0)

                if cos_sim > 0.65:  # threshold for recognition
                    st.session_state.authenticated = True
                    authenticated = True
                    break

        cap.release()
        login_placeholder.empty()  # remove the entire login section

    st.success("Face recognized! Presentation Begins...")
    time.sleep(1)

# -----------------------------
# After authentication
# -----------------------------
if st.session_state.authenticated:
    st.logo("combined_logo.png")

    pages = [
        ("page_0.py", "ML DP Presentation"),
        ("page_1.py", "Problem Description"),
        ("page_2.py", "Research Data"),
        ("page_3.py", "Features Engineering and Selection"),
        ("page_4.py", "ML & DL Model Selection"),
        ("page_5.py", "Final Model Prediction"),
        ("page_6.py", "Behind The Scenes"),
        ("page_7.py", "Conclusions")
    ]

    page_objs = [st.Page(path, title=title) for path, title in pages]
    pg = st.navigation(page_objs)
    pg.run()



#
#
# import base64
#
# import streamlit as st
# st.logo("combined_logo.png")
#
# page_0 = st.Page("page_0.py", title = "ML DP Presentation")
# page_1 = st.Page("page_1.py", title = "Problem Description")
# page_2 = st.Page("page_2.py", title = "Research Data")
# page_3 = st.Page("page_3.py", title = "Features Engineering and Selection")
# page_4 = st.Page("page_4.py", title = "ML & DL Model Selection")
# page_5 = st.Page("page_5.py", title = "Final Model Prediction")
# page_6 = st.Page("page_6.py", title = "Behind The Scenes")
# page_7 = st.Page("page_7.py", title = "Conclusions")
# # page_8 = st.Page("page_8.py", title = "Python Project Overview")
# # page_9 = st.Page("page_9.py", title = "Appreciation")
#
# # Set up navigation
# pg = st.navigation(
#     [page_0, page_1,page_2, page_3, page_4, page_5, page_6, page_7]
# )
#
# # run the selected page
# pg.run()

# 08-9170444
# 08-9268118
#