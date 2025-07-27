import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import tempfile
from fpdf import FPDF
import os

# ----- CONFIGURATION -----
st.set_page_config(page_title="🦷 Implant System Detection App", layout="centered")
st.title("🦷 Implant System Detection")
st.markdown("Upload an OPG/RVG image to detect implants using multiple models.")

# ---- Roboflow API Key ----
rf = Roboflow(api_key="4ZQ2GRG22mUeqtXFX26n")  # Replace with your actual API key
project = rf.workspace("implant-system-identification").project("implant-system-detection")
model_v7 = project.version(7).model
model_v8 = project.version(8).model
model_v4 = project.version(4).model

# ----- IMAGE UPLOAD -----
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        img_path = temp.name

    st.markdown("### Predictions:")
    predictions = []

    for name, model in zip(["YOLOv7 RF DETR", "YOLOv11 v8", "YOLOv8 v4"], [model_v7, model_v8, model_v4]):
        result = model.predict(img_path, confidence=40, overlap=30).json()
        pred_image = image.copy()
        draw = ImageDraw.Draw(pred_image)

        st.subheader(f"Model: {name}")

        for pred in result['predictions']:
            x0 = int(pred['x'] - pred['width'] / 2)
            y0 = int(pred['y'] - pred['height'] / 2)
            x1 = int(pred['x'] + pred['width'] / 2)
            y1 = int(pred['y'] + pred['height'] / 2)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0, y0 - 10), f"{pred['class']} ({pred['confidence']*100:.1f}%)", fill="red")

        st.image(pred_image, caption=f"Predictions from {name}", use_container_width=True)
        predictions.append((name, pred_image))

    # ----- PDF GENERATION -----
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="🦷 Dental Implant Detection Report", ln=True, align='C')
        pdf.ln(10)

        for model_name, pred_img in predictions:
            # Save prediction image to temp
            pred_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            pred_img.save(pred_path)

            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=model_name, ln=True, align='L')
            pdf.image(pred_path, w=pdf.w * 0.7)
            pdf.ln(10)

        # Contact Section
        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Created by Dr Balaganesh P", ln=True, align='C')
        pdf.cell(200, 10, txt="LinkedIn: https://www.linkedin.com/in/drbalaganeshdentist/", ln=True, align='C')
        pdf.cell(200, 10, txt="Instagram: https://www.instagram.com/_bala.7601/", ln=True, align='C')
        pdf.cell(200, 10, txt="Gmail: drbalaganesh.dentist@gmail.com", ln=True, align='C')

        # Save and show download link
        pdf_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        pdf.output(pdf_output_path)
        with open(pdf_output_path, "rb") as f:
            st.download_button(label="📄 Download PDF Report", data=f, file_name="Implant_Report.pdf")

# ----- CONTACT AND CREDITS SECTION -----
st.markdown("---")
st.markdown("### 📬 Contact")
st.markdown("""
<p style='text-align: center;'>
    <a href="mailto:drbalaganesh.dentist@gmail.com" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="30"></a>
    <a href="https://github.com/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30"></a>
    <a href="https://www.linkedin.com/in/drbalaganeshdentist/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30"></a>
    <a href="https://www.instagram.com/_bala.7601/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30"></a>
</p>
<p style='text-align: center;'>Created by Dr Balaganesh P</p>
""", unsafe_allow_html=True)
