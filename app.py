import streamlit as st
from roboflow import Roboflow
from PIL import Image
import pandas as pd
import datetime
import os
import io
import base64
from google.oauth2 import service_account
import gspread
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Streamlit config
st.set_page_config(page_title="Multi-Model Implant Detection", layout="centered")
st.title("🦷 Multi-Model Dental Implant Detection")

# ---- Authenticate Google Sheet and Drive ----
credentials_dict = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file"
    ]
)

# Authorize Google Sheets
gc = gspread.authorize(credentials)

# Build Drive service
drive_service = build('drive', 'v3', credentials=credentials)

# Google Sheet setup
sheet_name = "Implant log"
spreadsheet = gc.open(sheet_name)
worksheet = spreadsheet.sheet1

# Roboflow models
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
workspace = "implant-system-identification"
project_name = "implant-system-detection"

model_v7 = rf.workspace(workspace).project(project_name).version(7).model  # RFDETR
model_v8 = rf.workspace(workspace).project(project_name).version(8).model  # YOLOv11
model_v4 = rf.workspace(workspace).project(project_name).version(4).model  # YOLOv8

# ---- Upload image ----
uploaded_file = st.file_uploader("Upload an OPG/RVG image", type=["jpg", "jpeg", "png"])

def upload_to_drive(file_path, filename, folder_id):
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, mimetype='image/jpeg')
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    return file.get('id')

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Form Inputs
    with st.form("input_form"):
        dentist_prediction = st.text_input("Dentist Predicted Implant System")
        placement_date = st.date_input("Implant Placement Date", datetime.date.today())
        consent = st.checkbox("Patient consents to use this image for research purposes")
        submit_btn = st.form_submit_button("Run Models and Log Data")

    if submit_btn:
        with st.spinner("Analyzing with all models..."):
            # Save temp image
            image_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            temp_path = "/tmp/temp_img.jpg"
            img.save(temp_path)

            # Run predictions
            pred_v7 = model_v7.predict(temp_path).json()
            pred_v8 = model_v8.predict(temp_path).json()
            pred_v4 = model_v4.predict(temp_path).json()

            # Extract labels
            def extract_labels(pred):
                return ", ".join([p["class"] for p in pred.get("predictions", [])]) or "No detections"

            pred_labels_v7 = extract_labels(pred_v7)
            pred_labels_v8 = extract_labels(pred_v8)
            pred_labels_v4 = extract_labels(pred_v4)

            # Upload image to Google Drive
            drive_folder_id = st.secrets["drive_folder_id"]
            uploaded_file_id = upload_to_drive(temp_path, uploaded_file.name, drive_folder_id)

            # Log to Google Sheet
            row = [
                uploaded_file.name,
                str(datetime.datetime.now()),
                dentist_prediction,
                str(placement_date),
                "Yes" if consent else "No",
                pred_labels_v7,
                pred_labels_v8,
                pred_labels_v4,
                uploaded_file_id  # Optionally log the Drive file ID
            ]
            worksheet.append_row(row)
            st.success("✅ Data logged and image saved to Drive.")
