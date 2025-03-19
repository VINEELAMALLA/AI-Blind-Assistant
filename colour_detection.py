import cv2
import numpy as np
from ultralytics import YOLO

def get_dominant_color(image):
    if image is None or image.size == 0:
        return "Unknown"
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv_image.reshape((-1, 3))
    filtered_pixels = pixels[pixels[:, 2] > 50]
    
    if len(filtered_pixels) == 0:
        return "Unknown"
    
    unique, counts = np.unique(filtered_pixels, axis=0, return_counts=True)
    dominant_color = unique[np.argmax(counts)]
    dominant_rgb = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_HSV2BGR)[0][0]
    return get_color_name(dominant_rgb[2], dominant_rgb[1], dominant_rgb[0])

def get_color_name(R, G, B):
    colors = {
        "Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Magenta": (255, 0, 255),
        "White": (255, 255, 255), "Black": (0, 0, 0), "Gray": (128, 128, 128),
        "Orange": (255, 165, 0), "Purple": (128, 0, 128), "Pink": (255, 192, 203), "Brown": (139, 69, 19)
    }
    min_distance = float("inf")
    color_name = "Unknown"
    for name, (r, g, b) in colors.items():
        distance = np.sqrt((R - r) ** 2 + (G - g) ** 2 + (B - b) ** 2)
        if distance < min_distance:
            min_distance = distance
            color_name = name
    return color_name

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            
            object_region = frame[y1:y2, x1:x2]
            object_color = get_dominant_color(object_region)
            label_text = f"{label} - Color: {object_color}"
            
            if label.lower() == "person":
                upper_body = frame[y1:int(y1 + (y2 - y1) * 0.4), x1:x2]
                lower_body = frame[int(y1 + (y2 - y1) * 0.6):y2, x1:x2]
                person_color = get_dominant_color(upper_body)
                dress_color = get_dominant_color(lower_body)
                label_text = f"{label} - Skin: {person_color}, Dress: {dress_color}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Object Detection with Color", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()