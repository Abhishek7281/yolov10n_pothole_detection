import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from utils.detect import detect_potholes  # Import the detect function

st.set_page_config(page_title="YOLOv10 Pothole Detection", layout="wide")
st.title("🛣️ YOLOv10 Pothole Detection System")

uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file is not None:
    is_video = uploaded_file.type.startswith("video/")

    if is_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

        video = cv2.VideoCapture(temp_video_path)
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = video.read()
            if not ret:
                break
            detected_frame = detect_potholes(frame)
            out.write(detected_frame)

        video.release()
        out.release()
        st.video(output_video_path)

        with open(output_video_path, "rb") as file:
            st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

        os.remove(temp_video_path)
        os.remove(output_video_path)

    else:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        detected_img = detect_potholes(img_array)
        detected_pil = Image.fromarray(detected_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width=500)
        with col2:
            st.image(detected_pil, caption="Detected Potholes", width=500)

        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        detected_pil.save(temp_image_path)

        with open(temp_image_path, "rb") as file:
            st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

        os.remove(temp_image_path)
