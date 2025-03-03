import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model_path = "model path"  # Update path if needed
model = YOLO(model_path).to(device)

# Set video input path (change to 0 for webcam)
video_path = "video path"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # Run YOLOv8 inference
    results = model(frame, conf=0.3)

    for result in results:
        boxes = result.boxes.data.cpu().numpy()  # Convert tensor to numpy
        for box in boxes:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            label = f"Plate: {conf:.2f}"

            # Draw bounding box around number plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Extract number plate
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue  # Skip empty detections

            # Resize to zoom (adjust zoom factor if needed)
            zoom_factor = 3  # Increase for larger zoom
            zoomed_plate = cv2.resize(plate_crop, (plate_crop.shape[1] * zoom_factor, plate_crop.shape[0] * zoom_factor))

            # Calculate position to place zoomed plate above original plate
            zoom_h, zoom_w, _ = zoomed_plate.shape
            display_x1 = x1
            display_y1 = max(y1 - zoom_h - 10, 0)  # Ensure it stays within frame

            # Overlay zoomed plate onto the frame
            if display_y1 + zoom_h < frame.shape[0] and display_x1 + zoom_w < frame.shape[1]:
                frame[display_y1:display_y1 + zoom_h, display_x1:display_x1 + zoom_w] = zoomed_plate

    # Show output frame
    cv2.imshow("Number Plate Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
