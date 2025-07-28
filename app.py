import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageOps
import tempfile
from fpdf import FPDF
import pandas as pd
import os
import datetime

# --------- PAGE CONFIG -----------
st.set_page_config(page_title="Multi-Model Implant Detection", layout="wide")
st.title("🦷 Multi-Model Dental Implant Detection")
st.markdown("Upload an OPG/RVG image to detect implants using three models: YOLOv8, YOLOv11, and RFDETR")

# --------- ROBOTFLOW API INIT -----------
rf = Roboflow(api_key="4ZQ2GRG22mUeqtXFX26n")  # Replace with your API key

# --------- ROBOTFLOW MODELS -----------
project = rf.workspace("implant-system-identification").project("implant-system-detection")
model_v7 = project.version(7).model  # RFDETR
model_v8 = project.version(8).model  # YOLOv11
model_v4 = project.version(4).model  # YOLOv8

# --------- FILE UPLOAD -----------
uploaded_file = st.file_uploader("Upload an OPG/RVG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Save original image to temp path
    orig_path = os.path.join(tempfile.gettempdir(), "original.jpg")
    image.save(orig_path)

    # Display original
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection on all 3 models
    with st.spinner("Detecting implants..."):
        pred_v4 = model_v4.predict(orig_path, confidence=40, overlap=30).plot()
        pred_v7 = model_v7.predict(orig_path, confidence=40, overlap=30).plot()
        pred_v8 = model_v8.predict(orig_path, confidence=40, overlap=30).plot()

    # Display side-by-side
    st.markdown("### Detection Results")
    col1, col2, col3 = st.columns(3)
    col1.image(pred_v4, caption="YOLOv8")
    col2.image(pred_v8, caption="YOLOv11")
    col3.image(pred_v7, caption="RFDETR")

    # ----------- PDF GENERATION ------------
    if st.button("Generate PDF Report"):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save images
            original_path = os.path.join(temp_dir, "original.jpg")
            pred4_path = os.path.join(temp_dir, "yolov8.jpg")
            pred8_path = os.path.join(temp_dir, "yolov11.jpg")
            pred7_path = os.path.join(temp_dir, "rfdetr.jpg")

            image.save(original_path)
            pred_v4.save(pred4_path)
            pred_v8.save(pred8_path)
            pred_v7.save(pred7_path)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Implant Detection Report", ln=True, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 10, f"Generated on: {datetime.datetime.now()}", ln=True)

            # Original image
            pdf.cell(200, 10, "Original Image:", ln=True)
            pdf.image(original_path, w=150)
            pdf.ln(5)

            # Predictions
            pdf.cell(200, 10, "YOLOv8 Prediction:", ln=True)
            pdf.image(pred4_path, w=150)
            pdf.ln(5)

            pdf.cell(200, 10, "YOLOv11 Prediction:", ln=True)
            pdf.image(pred8_path, w=150)
            pdf.ln(5)

            pdf.cell(200, 10, "RFDETR Prediction:", ln=True)
            pdf.image(pred7_path, w=150)

            # Save final PDF
            pdf_path = os.path.join(temp_dir, "implant_report.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Report", f, file_name="implant_report.pdf")




st.markdown("""
    <style>
    .footer-container {
        margin-top: 50px;
        text-align: center;
        width: 100%;
    }
    .footer-text {
        font-weight: bold;
        color: #444;
        margin-bottom: 8px;
    }
    .footer-icons {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }
    .footer-icons a img {
        width: 30px;
        transition: transform 0.2s;
    }
    .footer-icons a img:hover {
        transform: scale(1.2);
    }
    .block-container {
        padding-bottom: 80px;
    }
    </style>
    <div class="footer-container">
        <p class="footer-text">Created by Dr Balaganesh P</p>
        <div class="footer-icons">
            <a href="mailto:drbalaganesh.dentist" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png">
            </a>
            <a href="https://github.com/DrDataScience-dentist" target="_blank">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png">
            </a>
            <a href="https://www.linkedin.com/in/drbalaganeshdentist/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
            </a>
            <a href="https://www.instagram.com/_bala.7601/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png">
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)
