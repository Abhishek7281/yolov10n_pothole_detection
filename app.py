# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# from utils.detect import detect_potholes  # Import the detect function

# st.set_page_config(page_title="YOLOv10 Pothole Detection", layout="wide")
# st.title("🛣️ YOLOv10 Pothole Detection System")

# uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

# if uploaded_file is not None:
#     is_video = uploaded_file.type.startswith("video/")

#     if is_video:
#         temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         temp_video.write(uploaded_file.read())
#         temp_video_path = temp_video.name

#         video = cv2.VideoCapture(temp_video_path)
#         output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         fps = int(video.get(cv2.CAP_PROP_FPS))
#         frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
#             detected_frame = detect_potholes(frame)
#             out.write(detected_frame)

#         video.release()
#         out.release()
#         st.video(output_video_path)

#         with open(output_video_path, "rb") as file:
#             st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#         os.remove(temp_video_path)
#         os.remove(output_video_path)

#     else:
#         image = Image.open(uploaded_file)
#         img_array = np.array(image)
#         detected_img = detect_potholes(img_array)
#         detected_pil = Image.fromarray(detected_img)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption="Original Image", width=500)
#         with col2:
#             st.image(detected_pil, caption="Detected Potholes", width=500)

#         temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#         detected_pil.save(temp_image_path)

#         with open(temp_image_path, "rb") as file:
#             st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#         os.remove(temp_image_path)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import torch
from ultralytics import YOLO

# Load YOLOv10n Model
@st.cache_resource()  # Cache model for faster inference
def load_model():
    model_path = "project_files/best.pt"  # Updated to best.pt for YOLOv10n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    return model

model = load_model()

# Pothole Detection Function
def detect_potholes(image):
    results = model(image)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Streamlit UI
def main():
    st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
    st.title("🛣️ YOLOv10n Pothole Detection System")

    uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file is not None:
        is_video = uploaded_file.type.startswith("video/")

        if is_video:
            try:
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

                progress_bar = st.progress(0)
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0

                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    detected_frame = detect_potholes(frame)
                    out.write(detected_frame)
                    frame_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                video.release()
                out.release()
                st.success("✅ Video processing complete!")

                # Display Processed Video
                st.video(output_video_path)

                # Allow Download
                with open(output_video_path, "rb") as file:
                    st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

                os.remove(temp_video_path)
                os.remove(output_video_path)

            except Exception as e:
                st.error(f"❌ Error processing video: {e}")

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

if __name__ == "__main__":
    main()
