import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cattle-Buffalo Recognition AI",
    layout="wide",
    page_icon="🐄"
)

# ---------------- PREMIUM CSS ----------------
premium_css = """
<style>

body {
    font-family: 'Segoe UI', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #6F00FF 0%, #9D50BB 100%);
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 40px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

.glass-card {
    background: rgba(255,255,255,0.20);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 30px;
}

.result-box {
    background: rgba(0,0,0,0.60);
    color: white;
    padding: 20px;
    border-radius: 20px;
}

</style>
"""
st.markdown(premium_css, unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown(
    """
    <div class="hero">
        <h1 style="font-size: 48px; font-weight: 800;">🐄 Cattle–Buffalo Recognition AI</h1>
        <p style="font-size: 20px; margin-top: -10px;">
            Upload an image and let the trained YOLOv8 model identify it instantly.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- DOWNLOAD MODEL FROM GOOGLE DRIVE ----------------

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?export=download&id=1zcF9zl_TgGFtvrsUuQaMAdMFLLzEvTUg"
    model_path = "model_temp.pt"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            r = requests.get(url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    return YOLO(model_path)

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📤 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a cattle/buffalo image", type=["jpg", "jpeg", "png"])
st.sidebar.info("Supported formats: JPG, PNG")

# ---------------- MAIN CONTENT ----------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📸 Uploaded Image")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)
    else:
        st.write("Upload an image from the sidebar.")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔍 Prediction Result")

    if uploaded_file:
        results = model.predict(img)
        pred = results[0].probs

        class_id = int(np.argmax(pred.data))
        confidence = float(pred.data[class_id]) * 100

        st.markdown(
            f"""
            <div class='result-box'>
                <h2 style="text-align:center;">RESULT</h2>
                <h1 style="text-align:center; font-size: 40px;">{model.names[class_id]}</h1>
                <h3 style="text-align:center;">Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Waiting for an image...")

    st.markdown("</div>", unsafe_allow_html=True)
