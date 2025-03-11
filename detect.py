from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv10 Nano model
model_path = "project_files/best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)

def detect_potholes(image):
    """
    Runs YOLOv10n model on an input image and returns the detected output.
    """
    results = model(image)  # Run inference

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image
