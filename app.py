import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import tempfile
from fpdf import FPDF
import os
import requests
from io import BytesIO

# --------- PAGE CONFIG -----------
st.set_page_config(page_title="🦷 Implant System Detection", layout="wide")
st.title("🦷 Implant System Detection System")
st.markdown("Upload an OPG/RVG image to detect implants using three different AI models.")

# ---------- ROBOLFLOW SETUP -----------
rf = Roboflow(api_key="4ZQ2GRG22mUeqtXFX26n")
project = rf.workspace("implant-system-identification").project("implant-system-detection")

models = {
    "YOLOv11 (v8)": project.version(8).model,
    "YOLOv8 (v4)": project.version(4).model,
    "RF-DETR (v7)": project.version(7).model,
}

# ---------- IMAGE UPLOAD -----------
uploaded_images = st.file_uploader("Upload OPG or RVG Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

results_data = []
if uploaded_images:
    for uploaded_image in uploaded_images:
        st.subheader(f"Results for {uploaded_image.name}")

        img = Image.open(uploaded_image).convert("RGB")
        buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(buffered.name)

        cols = st.columns(len(models))
        for idx, (model_name, model) in enumerate(models.items()):
            with cols[idx]:
                st.markdown(f"**{model_name}**")
                pred = model.predict(buffered.name, confidence=40, overlap=30).json()

                draw = ImageDraw.Draw(img.copy())
                output_img = img.copy()

                for detection in pred['predictions']:
                    x1 = detection['x'] - detection['width'] / 2
                    y1 = detection['y'] - detection['height'] / 2
                    x2 = detection['x'] + detection['width'] / 2
                    y2 = detection['y'] + detection['height'] / 2

                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1 - 10), f"{detection['class']} ({detection['confidence']*100:.1f}%)", fill="red")

                st.image(output_img, caption="Predictions", use_container_width=True)

                # Collect data for PDF
                results_data.append({
                    "image_name": uploaded_image.name,
                    "model_name": model_name,
                    "predictions": pred['predictions']
                })

# ---------- PDF EXPORT -----------
if results_data:
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for result in results_data:
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Implant System Detection Report", ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Image: {result['image_name']}", ln=True)
            pdf.cell(200, 10, txt=f"Model: {result['model_name']}", ln=True)
            pdf.ln(5)
            for pred in result['predictions']:
                label = pred['class']
                confidence = pred['confidence'] * 100
                pdf.cell(200, 10, txt=f"Prediction: {label}, Confidence: {confidence:.1f}%", ln=True)

        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Created by Dr. Balaganesh", ln=True)
        pdf.cell(200, 10, txt="Gmail: drbalaganesh.dentist@gmail.com", ln=True)
        pdf.cell(200, 10, txt="LinkedIn: https://www.linkedin.com/in/drbalaganeshdentist/", ln=True)
        pdf.cell(200, 10, txt="Instagram: https://www.instagram.com/_bala.7601/", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_file:
            pdf.output(pdf_file.name)
            with open(pdf_file.name, "rb") as f:
                st.download_button("Download PDF", f, file_name="Implant_Report.pdf")

# ---------- CONTACT SECTION -----------
st.markdown("---")
st.markdown("### 📬 Connect with Me")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("[![Gmail](https://img.icons8.com/fluency/48/gmail.png)](mailto:drbalaganesh.dentist@gmail.com)", unsafe_allow_html=True)
with col2:
    st.markdown("[![GitHub](https://img.icons8.com/ios-filled/50/github.png)](https://github.com/DrDataScience-dentist)", unsafe_allow_html=True)
with col3:
    st.markdown("[![LinkedIn](https://img.icons8.com/color/48/linkedin.png)](https://www.linkedin.com/in/drbalaganeshdentist/)", unsafe_allow_html=True)
with col4:
    st.markdown("[![Instagram](https://img.icons8.com/color/48/instagram-new.png)](https://www.instagram.com/_bala.7601/)", unsafe_allow_html=True)

st.markdown("<style>img { height: 35px !important; }</style>", unsafe_allow_html=True)
