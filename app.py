import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import tempfile
from fpdf import FPDF
import requests
from io import BytesIO
import base64
import os

# Set up Roboflow access
rf = Roboflow(api_key="4ZQ2GRG22mUeqtXFX26n")  # Replace with your actual API key
project = rf.workspace("implant-system-identification").project("implant-system-detection")

# Models
models = {
    "RF-DETR (v7)": project.version(7).model,
    "YOLOv11 (v8)": project.version(8).model,
    "YOLOv8 (v4)": project.version(4).model
}

# Streamlit setup
st.set_page_config(layout="wide", page_title="🦷 Implant System Detection")
st.title("🦷 IMPLANT SYSTEM DETECTION SYSTEM")
st.markdown("Upload a radiograph to run inference with RF-DETR, YOLOv11, and YOLOv8 models")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# Confidence meter helper
def draw_confidence_meter(confidence):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(2.5, 1.3))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.2, 1.2)
    theta = confidence / 100 * np.pi
    ax.plot([0, np.cos(theta - np.pi/2)], [0, np.sin(theta - np.pi/2)], lw=3, color='green')
    ax.axis('off')
    ax.set_title(f"Confidence: {confidence:.1f}%", fontsize=10)
    return fig

# Display and run model
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((512, 512))
    st.image(image, caption="Uploaded Image", use_column_width=False)

    results_pdf = FPDF()
    results_pdf.add_page()
    results_pdf.set_font("Arial", "B", 14)
    results_pdf.cell(0, 10, "Implant System Detection Report", ln=True)

    for model_name, model in models.items():
        st.subheader(f"🔍 Results from {model_name}")
        # Save image to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            prediction = model.predict(temp.name, confidence=40, overlap=30).json()

        draw = ImageDraw.Draw(image.copy())
        confidence_section = st.empty()
        for i, obj in enumerate(prediction["predictions"]):
            x1, y1 = obj["x"] - obj["width"]/2, obj["y"] - obj["height"]/2
            x2, y2 = obj["x"] + obj["width"]/2, obj["y"] + obj["height"]/2
            label = obj["class"]
            conf = obj["confidence"] * 100
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1-10), f"{label} ({conf:.1f}%)", fill="red")

            # Confidence meter
            col1, col2, col3 = st.columns(3)
            with [col1, col2, col3][i % 3]:
                st.pyplot(draw_confidence_meter(conf))

            results_pdf.set_font("Arial", "", 12)
            results_pdf.cell(0, 10, f"{model_name} → {label}: {conf:.2f}%", ln=True)

        st.image(image, caption=f"{model_name} Predictions", use_column_width=False)

    # Save PDF to download
    pdf_path = os.path.join(tempfile.gettempdir(), "implant_report.pdf")
    results_pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="implant_detection_report.pdf">📄 Download PDF Report</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

# Footer with icons
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>Created by Dr. Balaganesh</h4>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.markdown(
        """
        <div style="text-align: center;">
            <a href="https://www.linkedin.com/in/drbalaganeshdentist/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30">
            </a>
            &nbsp;&nbsp;
            <a href="https://www.instagram.com/_bala.7601/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30">
            </a>
        </div>
        """, unsafe_allow_html=True
    )
