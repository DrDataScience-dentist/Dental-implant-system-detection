import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
from fpdf import FPDF
import pandas as pd
import tempfile
import os
import urllib.request
import base64
import cv2

# --------- PAGE CONFIG -----------
st.set_page_config(page_title="🦷 Multi-Model Implant Detection", layout="wide")

st.write("OpenCV version:", cv2.__version__)

st.markdown(
    """
    <style>
    .main .block-container {
        padding: 0;
    }
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
model_v7 = rf.workspace("implant-system-identification").project("implant-system-detection").version(7).model
model_v8 = rf.workspace("implant-system-identification").project("implant-system-detection").version(8).model
model_v4 = rf.workspace("implant-system-identification").project("implant-system-detection").version(4).model

# --------- SIDEBAR SETTINGS ---------
st.sidebar.header("🔧 Prediction Settings")
confidence = st.sidebar.slider("Confidence Threshold (%)", min_value=10, max_value=90, value=40, step=5)
overlap = st.sidebar.slider("Overlap Threshold (%)", min_value=0, max_value=50, value=30, step=5)

# --------- PREDICTION FUNCTION ---------
def predict_and_draw(model, image_path, tag):
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    predictions = result['predictions']

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    data = []

    for i, pred in enumerate(predictions):
        class_name = pred['class']
        confidence_score = round(pred['confidence'] * 100, 2)
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        xmin = x - width / 2
        ymin = y - height / 2
        xmax = x + width / 2
        ymax = y + height / 2

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), f"{class_name} ({confidence_score}%)", fill="red")

        box_img = img.crop((xmin, ymin, xmax, ymax))
        temp_box = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        box_img.save(temp_box.name)

        data.append({
            "Class": class_name,
            "Confidence (%)": confidence_score,
            "Image Path": temp_box.name
        })

    return img, data

# --------- UPLOAD IMAGE ---------
uploaded_file = st.file_uploader("Upload your OPG/RVG image", type=["jpg", "jpeg", "png"])

# --------- MAIN DISPLAY ---------
if uploaded_file:
    image = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    # Convert image to base64
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()

    # Display centered image with caption
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{img_base64}" width="200">
            <p><em>Uploaded Image</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔹 RF DETR")
        pred_img_v7, data_v7 = predict_and_draw(model_v7, image_path, "RF")
        st.image(pred_img_v7, caption="RF DETR Prediction", width=200)
        st.dataframe(pd.DataFrame(data_v7).drop(columns=["Image Path"]))

    with col2:
        st.subheader("🔶 YOLOv11")
        pred_img_v8, data_v8 = predict_and_draw(model_v8, image_path, "YOLOv11")
        st.image(pred_img_v8, caption="YOLOv11 Prediction", width=200)
        st.dataframe(pd.DataFrame(data_v8).drop(columns=["Image Path"]))

    with col3:
        st.subheader("🔴 YOLOv8")
        pred_img_v4, data_v4 = predict_and_draw(model_v4, image_path, "YOLOv8")
        st.image(pred_img_v4, caption="YOLOv8 Prediction", width=200)
        st.dataframe(pd.DataFrame(data_v4).drop(columns=["Image Path"]))

    # --------- PDF GENERATION ---------
    if st.button("Generate PDF Report"):
        pdf = FPDF()

        # --------- Header Page ---------
       # --------- Header Page ---------
        pdf.add_page()
        header_url = "https://raw.githubusercontent.com/DrDataScience-dentist/Dental-implant-system-detection/main/header_pdf.png"
        header_path = os.path.join(tempfile.gettempdir(), "header.png")
        urllib.request.urlretrieve(header_url, header_path)
        pdf.image(header_path, x=10, y=10, w=190)
        
        # Title Section
        pdf.set_y(150)
        pdf.set_font("Courier", style='B', size=16)
        pdf.cell(pdf.w - 2 * pdf.l_margin, 10, txt="IMPLANT SYSTEM DETECTION REPORT", ln=True, align='C')
        


        def add_each_implant(title, data):
            title_clean = title.encode("ascii", "ignore").decode()
            for item in data:
                pdf.add_page()
        
                # Title
                pdf.set_font("Courier", style='B', size=14)
                pdf.cell(190, 10, txt=title_clean, ln=True, align='C')
                pdf.ln(10)
        
                # Image section
                if os.path.exists(item["Image Path"]):
                    img_width = 120  # set fixed width
                    page_width = 210  # A4 width in mm
                    x_center = (page_width - img_width) / 2  # center horizontally
                    pdf.image(item["Image Path"], x=x_center, w=img_width)
                    pdf.ln(5)
        
                # Text below image
                pdf.set_font("Arial", size=12)
                pdf.cell(190, 10, txt=f"Class: {item['Class']}", ln=True, align='C')
                pdf.cell(190, 10, txt=f"Confidence: {item['Confidence (%)']}%", ln=True, align='C')
                pdf.ln(5)


        add_each_implant("RF DETR", data_v7)
        add_each_implant("YOLOv11", data_v8)
        add_each_implant("YOLOv8", data_v4)

        # --------- Footer Page ---------
        pdf.add_page()
        pdf.set_y(150)  # Move near bottom of the page
        pdf.set_font("Courier", style='B', size=8)
        pdf.cell(190, 10, txt="Created by Dr Balaganesh P", ln=True, align='C')
        pdf.ln(5)

        icons = {
            "Gmail": ("https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png", "mailto:drbalaganesh.dentist"),
            "GitHub": ("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", "https://github.com/DrDataScience-dentist"),
            "LinkedIn": ("https://cdn-icons-png.flaticon.com/512/174/174857.png", "https://www.linkedin.com/in/drbalaganeshdentist/"),
            "Instagram": ("https://cdn-icons-png.flaticon.com/512/2111/2111463.png", "https://www.instagram.com/_bala.7601/")
        }

        icon_size = 8  # Small icon size
        padding = 5
        total_width = len(icons) * (icon_size + padding)
        start_x = (210 - total_width) // 2
        y_pos = pdf.get_y()

        for i, (name, (url, link)) in enumerate(icons.items()):
            icon_path = os.path.join(tempfile.gettempdir(), f"{name}.png")
            urllib.request.urlretrieve(url, icon_path)
            x_pos = start_x + i * (icon_size + padding)
            pdf.image(icon_path, x=x_pos, y=y_pos, w=icon_size, h=icon_size)
            pdf.link(x=x_pos, y=y_pos, w=icon_size, h=icon_size, link=link)


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
