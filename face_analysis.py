import cv2
import torch
from deepface import DeepFace
import mediapipe as mp

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(frame_rgb)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue
            
            # Perform analysis using DeepFace
            try:
                analysis = DeepFace.analyze(face_roi, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender']
                emotion = analysis[0]['dominant_emotion']
                
                # Draw bounding box and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Age: {age}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f'Emotion: {emotion}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f'Error analyzing face: {e}')
    
    # Display the frame
    cv2.imshow('Real-Time Face Analysis', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()