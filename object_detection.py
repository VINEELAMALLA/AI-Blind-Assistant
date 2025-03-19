import os
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import queue
from ultralytics import YOLO

# Suppress TensorFlow and PyTorch warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # Speed up speech

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load YOLOv8 model
model = YOLO('yolov8l.pt').to(device)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Speech queue for real-time voice output
speech_queue = queue.Queue()

def speak():
    """ Continuously speaks queued messages in a separate thread. """
    while True:
        text = speech_queue.get()
        if text:
            if not speech_queue.empty():
                continue  # Skip if new detections exist (prevents lag)
            engine.say(text)
            engine.runAndWait()

# Start speech thread
speech_thread = threading.Thread(target=speak, daemon=True)
speech_thread.start()

previous_objects = {}  # Store previous object positions
size_history = {}  # Track size changes over time
movement_cooldown = {}  # Prevent repeated announcements

# Thresholds
FORWARD_BACKWARD_THRESHOLD_RATIO = 0.10  # 10% size change triggers movement
COOLDOWN_TIME = 10  # Frames to wait before announcing same movement

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Webcam closed. Stopping detection.")
        break  

    # Normalize the image
    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Run YOLOv8 detection
    results = model(frame, conf=0.4, iou=0.45)

    current_objects = {}  # Store objects detected in the current frame
    detected_labels = set()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            label = result.names[int(box.cls[0])]
            size = abs(x2 - x1) * abs(y2 - y1)  # Area-based size estimation

            # Store object position
            current_objects[label] = size

            # Keep track of past sizes
            if label not in size_history:
                size_history[label] = []
            size_history[label].append(size)
            if len(size_history[label]) > 5:
                size_history[label].pop(0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detected_labels.add(label)

    # Detect forward/backward movement
    movement_messages = []
    for label, size in current_objects.items():
        if label in previous_objects:
            prev_size = previous_objects[label]

            # Forward/backward movement using rolling average
            avg_size = sum(size_history[label]) / len(size_history[label])
            size_change = (size - avg_size) / avg_size  # Relative size change

            if abs(size_change) > FORWARD_BACKWARD_THRESHOLD_RATIO:
                direction = "closer" if size_change > 0 else "away"
                if movement_cooldown.get(label, 0) == 0:
                    movement_messages.append(f"{label} moved {direction}.")
                    movement_cooldown[label] = COOLDOWN_TIME

    # Decrease cooldown counters
    for label in movement_cooldown:
        if movement_cooldown[label] > 0:
            movement_cooldown[label] -= 1

    # Speak only new movements
    if movement_messages:
        movement_text = " ".join(movement_messages)
        print(f"Movement: {movement_text}")
        speech_queue.queue.clear()  # Clear previous speech queue
        speech_queue.put(movement_text)

    # Speak only when new objects appear
    if set(current_objects.keys()) != set(previous_objects.keys()):
        detected_text = ", ".join(detected_labels)
        print(f"Detected: {detected_text}")
        speech_queue.queue.clear()
        speech_queue.put(f"I can see: {detected_text}")

    # Update previous objects
    previous_objects = current_objects.copy()

    # Show output
    cv2.imshow('Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break  

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
