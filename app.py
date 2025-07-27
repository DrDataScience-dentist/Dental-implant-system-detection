# app.py
import streamlit as st
from roboflow import Roboflow
from PIL import Image
import tempfile

# --------- PAGE CONFIG -----------
st.set_page_config(page_title="Implant Detection", layout="wide")
st.title("🦷 Dental Implant Detection (via Roboflow)")

# --------- API SETUP ------------
rf = Roboflow(api_key="4ZQ2GRG22mUeqtXFX26n")  # Replace with your actual key
project = rf.workspace("implant-system-identification").project("implant-system-detection")
model = project.version("7").model  # e.g., version("1").model

# --------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload an OPG/RVG Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        prediction = model.predict(tmp.name, confidence=40, overlap=30).json()

    # --------- PREDICTION OUTPUT ----------
    st.subheader("Detected Implants:")
    for pred in prediction['predictions']:
        st.write(f"- **{pred['class']}** with confidence {pred['confidence']*100:.2f}%")

    # Optional: Show raw JSON
    with st.expander("Show Raw Output"):
        st.json(prediction)
