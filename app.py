import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import tempfile
from fpdf import FPDF
import pandas as pd

# --------- PAGE CONFIG -----------
st.set_page_config(page_title="Multi-Model Implant Detection", layout="wide")
st.title("🦷 Multi-Model Dental Implant Detection")
st.markdown("Upload an OPG/RVG image to detect implants using 4 different models trained on the same 7 classes.")

# --------- ROBFLOW CONFIG -----------
API_KEY = "4ZQ2GRG22mUeqtXFX26n"  # <-- Replace this with your API key

models_config = [
    {"name": "Roboflow v3 model", "project": "implant-system-detection", "version": 1},
    {"name": "YOLOv8 model", "project": "implant-system-detection", "version": 4},
    {"name": "rf-Detr model", "project": "implant-system-detection", "version": 7},
    {"name": "YOLOv11 model", "project": "implant-system-detection", "version": 8}
]

rf = Roboflow(api_key=API_KEY)

# --------- UPLOAD IMAGE -----------
uploaded_file = st.file_uploader("📤 Upload or Drag & Drop an RVG/OPG Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    annotated_images = []
    all_detections = []

    with st.spinner("Running detections on all models..."):
        for model_info in models_config:
            project = rf.workspace().project(model_info["project"])
            model = project.version(model_info["version"]).model

            # Save temp image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                image.save(tmp_file.name)
                pred = model.predict(tmp_file.name, confidence=40, overlap=30).json()

            # Draw results
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            detections = []

            for obj in pred["predictions"]:
                x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
                class_name = obj["class"]
                conf = obj["confidence"] * 100
                label = f"{class_name} ({conf:.1f}%)"
                draw.rectangle([(x - w / 2, y - h / 2), (x + w / 2, y + h / 2)], outline="red", width=2)
                draw.text((x - w / 2, y - h / 2 - 10), label, fill="red")
                detections.append(label)

            annotated_images.append((model_info["name"], draw_image, detections))
            all_detections.append((model_info["name"], detections))

    # --------- DISPLAY RESULTS -----------
    st.markdown("### 🧪 Annotated Detections from Each Model")
    cols = st.columns(4)
    for i, (model_name, img, dets) in enumerate(annotated_images):
        with cols[i]:
            st.image(img, caption=model_name, use_column_width=True)
            if dets:
                st.markdown("**Detections:**")
                for d in dets:
                    st.markdown(f"- {d}")
            else:
                st.markdown("_No detections._")

    # --------- COMPARISON TABLE -----------
    st.markdown("### 📊 Comparison Table of Predictions")
    table_data = []
    for model_name, dets in all_detections:
        for det in dets:
            label_split = det.split(" (")
            class_name = label_split[0]
            confidence = label_split[1].replace("%)", "") if len(label_split) > 1 else "0"
            table_data.append({
                "Model": model_name,
                "Implant Type": class_name,
                "Confidence (%)": float(confidence)
            })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df.sort_values(by="Confidence (%)", ascending=False), use_container_width=True)
    else:
        st.info("No implants were detected by any model.")

    # --------- PDF GENERATION -----------
    if st.button("📄 Download Report as PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Dental Implant Detection Report", ln=True, align="C")

        for model_name, dets in all_detections:
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(200, 10, txt=f"\nModel: {model_name}", ln=True)
            pdf.set_font("Arial", size=12)
            if dets:
                for det in dets:
                    pdf.cell(200, 10, txt=f" - {det}", ln=True)
            else:
                pdf.cell(200, 10, txt=" - No detections", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            st.success("✅ PDF Generated!")
            st.download_button(label="📥 Download PDF",
                               data=open(tmp_pdf.name, "rb"),
                               file_name="implant_detection_report.pdf",
                               mime="application/pdf")
