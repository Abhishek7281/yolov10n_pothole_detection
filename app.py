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

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# import torch
# from ultralytics import YOLO

# # Load YOLOv10n Model
# @st.cache_resource()  # Cache model for faster inference
# def load_model():
#     model_path = "project_files/best.pt"  # Updated to best.pt for YOLOv10n
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# model = load_model()

# # Pothole Detection Function
# def detect_potholes(image):
#     results = model(image)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return image

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
#     st.title("🛣️ YOLOv10n Pothole Detection System")

#     uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith("video/")

#         if is_video:
#             try:
#                 temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 temp_video.write(uploaded_file.read())
#                 temp_video_path = temp_video.name

#                 video = cv2.VideoCapture(temp_video_path)
#                 output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                 fps = int(video.get(cv2.CAP_PROP_FPS))
#                 frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#                 progress_bar = st.progress(0)
#                 total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frame_count = 0

#                 while True:
#                     ret, frame = video.read()
#                     if not ret:
#                         break
#                     detected_frame = detect_potholes(frame)
#                     out.write(detected_frame)
#                     frame_count += 1
#                     progress_bar.progress(min(frame_count / total_frames, 1.0))

#                 video.release()
#                 out.release()
#                 st.success("✅ Video processing complete!")

#                 # Display Processed Video
#                 st.video(output_video_path)

#                 # Allow Download
#                 with open(output_video_path, "rb") as file:
#                     st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#                 os.remove(temp_video_path)
#                 os.remove(output_video_path)

#             except Exception as e:
#                 st.error(f"❌ Error processing video: {e}")

#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array)
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", width=500)
#             with col2:
#                 st.image(detected_pil, caption="Detected Potholes", width=500)

#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)

#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st

# # ✅ `st.set_page_config()` must be the FIRST Streamlit command!
# st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")

# import asyncio
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# import torch
# from ultralytics import YOLO

# # ✅ Fix RuntimeError: no running event loop (Torch issue)
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.run(asyncio.sleep(0))

# # Load YOLOv10n Model
# @st.cache_resource()  # Cache model for faster inference
# def load_model():
#     model_path = "project_files/best.pt"  # ✅ Use best.pt for YOLOv10n
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# model = load_model()

# # Pothole Detection Function
# # def detect_potholes(image):
# #     results = model(image)

# #     for r in results:
# #         for box in r.boxes:
# #             x1, y1, x2, y2 = map(int, box.xyxy[0])
# #             confidence = float(box.conf[0])
# #             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# #     return image

# def detect_potholes(image):
#     results = model(image)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#             confidence = float(box.conf[0])  # Confidence score

#             # Draw thicker bounding box with deep color
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow box

#             # Create background for text
#             label = f"{confidence:.2f}"
#             (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
#             cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 255), -1)  # Filled bg

#             # Put text over the rectangle
#             cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)  # Black text

#     return image

# # Streamlit UI
# def main():
#     st.title("🛣️ YOLOv10n Pothole Detection System")
#     uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith("video/")

#         if is_video:
#             try:
#                 temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 temp_video.write(uploaded_file.read())
#                 temp_video_path = temp_video.name

#                 video = cv2.VideoCapture(temp_video_path)
#                 output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                 fps = int(video.get(cv2.CAP_PROP_FPS))
#                 frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#                 progress_bar = st.progress(0)
#                 total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frame_count = 0

#                 while True:
#                     ret, frame = video.read()
#                     if not ret:
#                         break
#                     detected_frame = detect_potholes(frame)
#                     out.write(detected_frame)
#                     frame_count += 1
#                     progress_bar.progress(min(frame_count / total_frames, 1.0))

#                 video.release()
#                 out.release()
#                 st.success("✅ Video processing complete!")

#                 # Display Processed Video
#                 st.video(output_video_path)

#                 # Allow Download
#                 with open(output_video_path, "rb") as file:
#                     st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#                 os.remove(temp_video_path)
#                 os.remove(output_video_path)

#             except Exception as e:
#                 st.error(f"❌ Error processing video: {e}")

#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array)
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", width=500)
#             with col2:
#                 st.image(detected_pil, caption="Detected Potholes", width=500)

#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)

#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

import streamlit as st

# ✅ `st.set_page_config()` must be the FIRST Streamlit command!
st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")

import asyncio
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import torch
from ultralytics import YOLO

# ✅ Fix RuntimeError: no running event loop (Torch issue)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# ✅ Load YOLOv10n Model
@st.cache_resource()  # Cache model for efficiency
def load_model():
    model_path = "project_files/best.pt"  # Path to YOLOv10n model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    return model

model = load_model()

# ✅ Pothole Detection Function
def detect_potholes(image):
    """
    Detect potholes using YOLOv10n and apply consistent bounding boxes and text.
    """
    results = model(image)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score

            # ✅ Define consistent styling
            bbox_color = (0, 255, 0)  # ✅ Green Bounding Box
            text_color = (0, 0, 0)  # Black Text
            thickness = 4  # Bold bounding box
            font_scale = 1.2
            font_thickness = 3

            # ✅ Draw Green bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thickness)

            # ✅ Create background for text
            label = f"{confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(image, (x1, y1 - text_height - 10), 
                          (x1 + text_width + 10, y1), bbox_color, -1)  # Filled bg

            # ✅ Add text over the rectangle
            cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, font_thickness, cv2.LINE_AA)

    return image

# ✅ Streamlit UI
def main():
    st.title("🛣️ YOLOv10n Pothole Detection System")
    uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file is not None:
        is_video = uploaded_file.type.startswith("video/")

        if is_video:
            try:
                # ✅ Save uploaded video
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name

                # ✅ Read video
                video = cv2.VideoCapture(temp_video_path)
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = int(video.get(cv2.CAP_PROP_FPS))
                frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                # ✅ Process video frames
                progress_bar = st.progress(0)
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0

                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    detected_frame = detect_potholes(frame_rgb)
                    detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)  # Convert back for OpenCV
                    out.write(detected_frame)

                    frame_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                # ✅ Release resources
                video.release()
                out.release()
                st.success("✅ Video processing complete!")

                # ✅ Display Processed Video
                st.video(output_video_path)

                # ✅ Allow Download
                with open(output_video_path, "rb") as file:
                    st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

                # ✅ Cleanup
                os.remove(temp_video_path)
                os.remove(output_video_path)

            except Exception as e:
                st.error(f"❌ Error processing video: {e}")

        else:
            # ✅ Process image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            detected_img = detect_potholes(img_array)
            detected_pil = Image.fromarray(detected_img)

            # ✅ Display Original & Processed Images
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", width=725)
            with col2:
                st.image(detected_pil, caption="Detected Potholes", width=725)

            # ✅ Save & Download Processed Image
            temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            detected_pil.save(temp_image_path)
            with open(temp_image_path, "rb") as file:
                st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

            # ✅ Cleanup
            os.remove(temp_image_path)

if __name__ == "__main__":
    main()
