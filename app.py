import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import tempfile
import os
from fpdf import FPDF
import pandas as pd

# ---------- CONFIG ----------
st.set_page_config(page_title="Multi-Model Implant Detection", layout="wide")
st.title("🦷 Multi-Model Dental Implant Detection")
st.markdown("Upload an OPG/RVG image to detect dental implants and generate a report.")

# ---------- SETUP ----------
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("your-project-name")
model = project.version("x").model

# ---------- FUNCTIONS ----------

def run_detection(image_path):
    result = model.predict(image_path, confidence=40, overlap=30).json()
    return result

def draw_boxes(image_path, predictions):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for pred in predictions['predictions']:
        x0 = pred['x'] - pred['width'] / 2
        y0 = pred['y'] - pred['height'] / 2
        x1 = pred['x'] + pred['width'] / 2
        y1 = pred['y'] + pred['height'] / 2
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0 - 10), pred['class'], fill="red")
    return img

def generate_pdf(predictions, image_path):
    from fpdf import FPDF

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.add_page()
            self.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
            self.set_font("DejaVu", "", 12)

    pdf = PDF()

    pdf.set_font("DejaVu", size=16)
    pdf.cell(0, 10, "🦷 Dental Implant Detection Report", ln=True)

    pdf.set_font("DejaVu", size=11)
    for pred in predictions['predictions']:
        pdf.cell(0, 10, f"{pred['class']} - Confidence: {pred['confidence']*100:.2f}%", ln=True)

    # Add the image
    pdf.image(image_path, x=10, y=None, w=180)

    # Footer
    pdf.set_y(-40)
    pdf.set_font("DejaVu", size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Created by Dr Balaganesh P", ln=True, align='C')
    
    pdf.set_text_color(0, 0, 255)
    pdf.cell(0, 10, "📧 drbalaganesh.dentist@gmail.com", ln=True, align='C', link="mailto:drbalaganesh.dentist@gmail.com")
    pdf.cell(0, 10, "🔗 LinkedIn", ln=True, align='C', link="https://linkedin.com/in/drbalaganeshdentist/")
    pdf.cell(0, 10, "📸 Instagram", ln=True, align='C', link="https://instagram.com/_bala.7601/")

    # Save to temporary path
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

# ---------- STREAMLIT UI ----------

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting implants..."):
        result = run_detection(tmp_path)
        output_img = draw_boxes(tmp_path, result)

    st.image(output_img, caption="Detection Result", use_column_width=True)

    pdf_path = generate_pdf(result, tmp_path)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Report PDF",
            data=f,
            file_name="implant_report.pdf",
            mime="application/pdf"
        )
