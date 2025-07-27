import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
from fpdf import FPDF
import tempfile
import os
import requests
from io import BytesIO

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="🦷 Implant Detection System", layout="wide")
st.title("🦷 Implant System Detection")
st.markdown("Detect dental implant systems from radiographs using 3 state-of-the-art models.")

# ----------------- PDF CLASS -----------------
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, "🦷 Implant System Detection Results", 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

# ----------------- INIT ROBOFLOW -----------------
rf = Roboflow(api_key="your_roboflow_api_key")
project = rf.workspace("implant-system-identification").project("implant-system-detection")
model7 = project.version(7).model  # RF-DETR
model8 = project.version(8).model  # YOLOv11
model4 = project.version(4).model  # YOLOv8
models = {
    "RF-DETR (v7)": model7,
    "YOLOv11 (v8)": model8,
    "YOLOv8 (v4)": model4
}

# ----------------- FILE UPLOAD -----------------
st.sidebar.header("Upload Images")
uploaded_files = st.sidebar.file_uploader("Choose OPG/RVG images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

# ----------------- RUN MODELS -----------------
if uploaded_files:
    selected_models = st.multiselect("Select Models to Run", options=list(models.keys()), default=list(models.keys()))

    image_results = []
    for uploaded_file in uploaded_files:
        st.subheader(f"Original: {uploaded_file.name}")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=350)

        for model_name in selected_models:
            model = models[model_name]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                image.save(temp.name)
                pred = model.predict(temp.name, confidence=40, overlap=30).json()

            pred_img = image.copy()
            draw = ImageDraw.Draw(pred_img)
            for det in pred['predictions']:
                x0 = det['x'] - det['width'] / 2
                y0 = det['y'] - det['height'] / 2
                x1 = det['x'] + det['width'] / 2
                y1 = det['y'] + det['height'] / 2
                label = det['class']
                conf = round(det['confidence'] * 100, 1)
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                draw.text((x0, y0 - 10), f"{label} ({conf}%)", fill="red")

            st.markdown(f"**Predicted with {model_name}:**")
            st.image(pred_img, width=350)
            image_results.append((uploaded_file.name, model_name, pred_img))

    # ----------------- PDF DOWNLOAD -----------------
    if st.button("📄 Download All Predictions as PDF"):
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for filename, model_name, img in image_results:
            temp_path = os.path.join(tempfile.gettempdir(), f"{filename}_{model_name}.jpg")
            img.save(temp_path)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, f"{model_name} - {filename}", ln=True)
            pdf.image(temp_path, w=180)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_file:
            pdf.output(pdf_file.name)
            st.success("PDF generated successfully!")
            st.download_button("⬇️ Download PDF", data=open(pdf_file.name, "rb"), file_name="implant_predictions.pdf")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("### Developed by Dr. Balaganesh")
st.markdown("Connect with me:")
st.markdown("""
<a href="https://www.linkedin.com/in/drbalaganeshdentist/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" style="margin-right: 10px"></a>
<a href="https://www.instagram.com/_bala.7601/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="25"></a>
""", unsafe_allow_html=True)
