import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
from fpdf import FPDF
import pandas as pd
import tempfile
import os

# --------- PAGE CONFIG -----------
st.set_page_config(page_title="🦷 Multi-Model Implant Detection", layout="wide")

st.markdown(
    """
    <style>
    /* Remove Streamlit's default padding */
    .main .block-container {
        padding: 0;
    }

    /* Full-width header image */
    .full-width-header {
        display: block;
        width: 100%;
        height: auto;
        margin: 0 auto;
    }
    </style>

    <img class="full-width-header" src="https://raw.githubusercontent.com/DrDataScience-dentist/Dental-implant-system-detection/main/header.png">
    """,
    unsafe_allow_html=True
)

# --------- ROBOFLOW INIT -----------
rf = Roboflow(api_key=st.secrets["roboflow"]["api_key"])

project_v7 = rf.workspace("implant-system-identification").project("implant-system-detection")
model_v7 = project_v7.version(7).model

project_v8 = rf.workspace("implant-system-identification").project("implant-system-detection")
model_v8 = project_v8.version(8).model

project_v4 = rf.workspace("implant-system-identification").project("implant-system-detection")
model_v4 = project_v4.version(4).model

# --------- SIDEBAR SETTINGS ---------
st.sidebar.header("🔧 Prediction Settings")
confidence = st.sidebar.slider("Confidence Threshold (%)", min_value=10, max_value=90, value=40, step=5)
overlap = st.sidebar.slider("Overlap Threshold (%)", min_value=0, max_value=50, value=30, step=5)

# --------- UPLOAD IMAGE ---------
uploaded_file = st.file_uploader("Upload your OPG/RVG image", type=["jpg", "jpeg", "png"])

# --------- PREDICTION FUNCTION ---------
def predict_and_draw(model, image_path):
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    predictions = result['predictions']

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    data = []

    for pred in predictions:
        class_name = pred['class']
        confidence_score = round(pred['confidence'] * 100, 2)
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        xmin = x - width / 2
        ymin = y - height / 2
        xmax = x + width / 2
        ymax = y + height / 2
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), f"{class_name} ({confidence_score}%)", fill="red")
        data.append({"Class": class_name, "Confidence (%)": confidence_score})

    return img, data

# --------- MAIN DISPLAY ---------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔷 RF DETR")
        pred_img_v7, data_v7 = predict_and_draw(model_v7, image_path)
        st.image(pred_img_v7, caption="RF DETR Prediction", use_container_width=True)
        st.dataframe(pd.DataFrame(data_v7))

    with col2:
        st.subheader("🔶 YOLOv11 - YOLOv8")
        pred_img_v8, data_v8 = predict_and_draw(model_v8, image_path)
        st.image(pred_img_v8, caption="YOLOv11 Prediction", use_container_width=True)
        st.dataframe(pd.DataFrame(data_v8))

    with col3:
        st.subheader("🔴 YOLOv8 - Original")
        pred_img_v4, data_v4 = predict_and_draw(model_v4, image_path)
        st.image(pred_img_v4, caption="YOLOv8 Prediction", use_container_width=True)
        st.dataframe(pd.DataFrame(data_v4))

    # --------- PDF GENERATION ---------
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        # Add header image (make sure width fits within A4 page — max ~190mm)
        pdf.image("https://raw.githubusercontent.com/DrDataScience-dentist/Dental-implant-system-detection/main/header_pdf.png", x=10, y=10, w=190)
        pdf.ln(35)
        # Adjust this value based on header image height
        pdf.set_font("Courier", style='B', size=14)
        pdf.cell(200, 10, txt="IMPLANT REPORT", ln=True, align='C')
        pdf.ln(10)

        def add_prediction_section(title, data):
            pdf.set_font("Courier", style='B', size=12)
            pdf.cell(200, 10, txt=title, ln=True)
            pdf.set_font("Arial", size=11)
            for item in data:
                pdf.cell(200, 10, txt=f"Class: {item['Class']}, Confidence: {item['Confidence (%)']}%", ln=True)
            pdf.ln(5)

        add_prediction_section("RF DETR", data_v7)
        add_prediction_section("YOLOv11 - YOLOv8", data_v8)
        add_prediction_section("YOLOv8 - Original", data_v4)

        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Contact: drbalaganesh.dentist", ln=True, link="mailto:drbalaganesh.dentist")
        pdf.cell(200, 10, txt="LinkedIn", ln=True, link="https://www.linkedin.com/in/drbalaganeshdentist/")
        pdf.cell(200, 10, txt="GitHub", ln=True, link="https://github.com/balaganesh7601")
        pdf.cell(200, 10, txt="Instagram", ln=True, link="https://www.instagram.com/_bala.7601/")
        pdf.cell(200, 10, txt="Created by Dr Balaganesh P", ln=True)

        pdf_output_path = os.path.join(tempfile.gettempdir(), "detection_report.pdf")
        pdf.output(pdf_output_path)

        with open(pdf_output_path, "rb") as f:
            st.download_button(label="📄 Download Report PDF", data=f, file_name="ImplantDetectionReport.pdf")

# --------- FOOTER ---------
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
