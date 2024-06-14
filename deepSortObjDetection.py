import cv2
import numpy as np
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize DeepSort object
deepsort = DeepSort()

# Open video file
cap = cv2.VideoCapture("sample_video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from the first frame

# Read the first frame
res, frame = cap.read()

# Select ROI in the first frame
region = cv2.selectROI("Select the area in the frame: ", frame)
cv2.destroyWindow("Select the area in the frame: ")

x, y, w, h = [int(a) for a in region]
roi = frame[y:y+h, x:x+w]

# Load YOLO object detector (using YOLOv3 in this example)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Object detection function
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], confidences[i]) for i in indices.flatten()]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Extract ROI from the frame
    roi_frame = frame[y:y+h, x:x+w]

    # Detect objects in the ROI
    detections = detect_objects(roi_frame)
    detections = [([x + box[0], y + box[1], box[2], box[3]], conf, 'object') for box, conf in detections]

    # Update DeepSort tracker with current frame detections
    tracks = deepsort.update_tracks(detections, frame=frame)

    # Draw tracked bounding boxes on the frame
    for track in tracks:
        bbox = track.to_tlbr()  # Get bounding box coordinates
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track.track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("result", frame)

    # Exit loop on 'q' key press
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
