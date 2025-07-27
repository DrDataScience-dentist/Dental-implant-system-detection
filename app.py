
import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import tempfile
from fpdf import FPDF
import os

# ---- PAGE CONFIG ----
st.set_page_config(page_title="🦷 Multi-Version Implant Detection", layout="wide")
st.title("🦷 Implant System Detection using Multiple Model Versions")
st.markdown("Upload an OPG/RVG image to detect implants using 3 versions of the same model.")

# ---- SETUP ROBOFLOW ----
rf = Roboflow(api_key="4ZQ2GRG22mUeqtXFX26n")  
project = rf.workspace("implant-system-identification").project("implant-system-detection")

# Define model versions
versions = {
    "Version 7": project.version(7).model,
    "Version 8": project.version(8).model,
    "Version 4": project.version(4).model,
}

# ---- UPLOAD IMAGE ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    base_image = Image.open(uploaded_file).convert("RGB")
    st.image(base_image, caption="Original Image", use_column_width=True)

    # Initialize PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for version_name, model in versions.items():
        st.subheader(version_name)

        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            base_image.save(tmp.name)
            prediction = model.predict(tmp.name, confidence=40, overlap=30).json()

        # Draw predictions
        img_with_boxes = base_image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        font = ImageFont.load_default()

        for pred in prediction['predictions']:
            x0 = pred['x'] - pred['width'] / 2
            y0 = pred['y'] - pred['height'] / 2
            x1 = pred['x'] + pred['width'] / 2
            y1 = pred['y'] + pred['height'] / 2

            draw.rectangle([x0, y0, x1, y1], outline="blue", width=3)
            label = f"{pred['class']} ({pred['confidence']*100:.1f}%)"
            draw.text((x0, y0 - 10), label, fill="blue", font=font)

        # Display image with predictions
        st.image(img_with_boxes, caption=f"{version_name} - Predictions", use_column_width=True)

        # Save image for PDF
        img_path = f"/tmp/{version_name.replace(' ', '_')}_result.jpg"
        img_with_boxes.save(img_path)

        # Add to PDF
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"{version_name} Predictions", ln=True, align='L')
        pdf.image(img_path, x=10, y=25, w=180)

    # Save PDF
    pdf_path = "/tmp/multiversion_implant_report.pdf"
    pdf.output(pdf_path)

    # Download Button
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download PDF Report", f, file_name="implant_predictions_report.pdf", mime="application/pdf")
