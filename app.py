import streamlit as st
from PIL import Image
from utils import classify_image, get_disposal_method
import time

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .stApp { max-width: 900px; margin: auto; font-family: 'Segoe UI', sans-serif; }
    .title { text-align: center; font-size: 32px; font-weight: bold; color: #2c3e50; margin-top: 2rem; }
    .subtitle { text-align: center; font-size: 16px; color: #7f8c8d; margin-bottom: 2rem; }
    .result-box { background-color: #f4f6f7; padding: 1.5rem; border-left: 5px solid #27ae60; border-radius: 8px; margin-top: 1.5rem; }
    .footer { margin-top: 3rem; text-align: center; font-size: 0.85rem; color: #95a5a6; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This Smart Waste Classifier uses a trained ML model to identify waste types:
    
    - 🟩 **General**
    - 🟧 **Hazardous**
    - 🟫 **Organic**
    - ♻️ **Recyclable**
    
    ### 🔧 How It Works
    1. Upload a waste image
    2. The image is resized & processed
    3. ML model predicts the waste type
    4. Get disposal suggestion instantly!
    """)

# Title & Subtitle
st.markdown("<div class='title'>♻️ Smart Waste Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a waste image to classify it and get a disposal tip.</div>", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image... please wait"):
        start = time.time()
        label = classify_image(image)
        elapsed = time.time() - start

    if label != "Classification error":
        disposal_tip = get_disposal_method(label)
        st.markdown(f"""
        <div class='result-box'>
            <h4>🗂️ Predicted Category: <b>{label}</b></h4>
            <p>🕒 Processed in {elapsed:.2f} seconds</p>
            <p>🗑️ <b>Disposal Tip:</b> {disposal_tip}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Could not classify this image. Try another one.")

# Footer
st.markdown("<div class='footer'>Smart Waste Classifier v2.0 • © 2025 WasteAI</div>", unsafe_allow_html=True)


