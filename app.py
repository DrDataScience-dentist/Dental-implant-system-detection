import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import tempfile
import pandas as pd
from datetime import datetime
import os
import json
import shutil

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- Streamlit Setup ---
st.set_page_config(page_title="🦷 Multi-Model Implant Detection", layout="wide")
st.title("🦷 Multi-Model Dental Implant Detection")
st.markdown("Upload an OPG or RVG image to detect implants using 3 models: RFDETR, YOLOv11, and YOLOv8.")

# --- Roboflow API Setup ---
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
project_v7 = rf.workspace("implant-system-identification").project("implant-system-detection")
model_v7 = project_v7.version(7).model  # RFDETR
model_v8 = project_v7.version(8).model  # YOLOv11
model_v4 = project_v7.version(4).model  # YOLOv8

# --- Google Credentials Setup ---
creds_dict = json.loads(st.secrets["gcp_service_account"])
creds = service_account.Credentials.from_service_account_info(creds_dict)
drive_service = build('drive', 'v3', credentials=creds)
sheets_service = build('sheets', 'v4', credentials=creds)

# --- Your Google Sheet and Drive Details ---
SHEET_NAME = "Implant log"
DRIVE_FOLDER_NAME = "implant images"
DRIVE_PARENT_PATH = "My Drive/Implant Upload"

# --- Helper: Upload Image to Google Drive ---
def upload_to_drive(file_path, file_name):
    folder_id = get_folder_id(DRIVE_FOLDER_NAME, DRIVE_PARENT_PATH)
    media = MediaFileUpload(file_path, resumable=True)
    file_metadata = {"name": file_name, "parents": [folder_id]}
    uploaded = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return uploaded.get("id")

def get_folder_id(folder_name, parent_path):
    response = drive_service.files().list(q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'", fields="files(id, name)").execute()
    folders = response.get("files", [])
    if folders:
        return folders[0]["id"]
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = drive_service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

# --- Helper: Log to Google Sheet ---
def log_to_sheet(row_data):
    sheet = sheets_service.spreadsheets()
    sheet_id = st.secrets["SHEET_ID"]
    sheet.values().append(
        spreadsheetId=sheet_id,
        range=f"{SHEET_NAME}!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": [row_data]}
    ).execute()

# --- Image Upload ---
uploaded_file = st.file_uploader("📤 Upload OPG/RVG Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        image_path = tmp_file.name
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Inputs ---
    dentist_prediction = st.text_input("🩺 Dentist Predicted Implant System")
    implant_date = st.date_input("📅 Implant Placement Date")
    consent = st.checkbox("✅ I consent to use this image for further research")

    # --- Predict Buttons ---
    if st.button("🔍 Predict using All Models"):
        st.markdown("### Model Results")

        for model, name in zip([model_v7, model_v8, model_v4], ["RFDETR (v7)", "YOLOv11 (v8)", "YOLOv8 (v4)"]):
            st.subheader(name)
            result = model.predict(image_path, confidence=40, overlap=30).json()
            predictions = result["predictions"]

            # Draw boxes
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for pred in predictions:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                class_name = pred["class"]
                draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=3)
                draw.text((x - w/2, y - h/2 - 10), class_name, fill="red")

            st.image(img, use_column_width=True)

        # --- Upload to Drive ---
        if consent:
            file_id = upload_to_drive(image_path, uploaded_file.name)

        # --- Log Data to Sheet ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_row = [timestamp, uploaded_file.name, str(implant_date), dentist_prediction, "Yes" if consent else "No"]
        log_to_sheet(log_row)

        st.success("✅ Detection Complete & Data Logged")

