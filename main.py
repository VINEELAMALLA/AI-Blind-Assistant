import cv2
import threading
import queue
import camera
import object_detection
import colour_detection
import face_analysis
import translation
import text2speech

# Queue to share frames and descriptions between threads
frame_queue = queue.Queue()
desc_queue = queue.Queue()

# Function to continuously capture frames from the camera
def capture_frames():
    cap = camera.open_camera()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        frame_queue.put(frame)  # Send frame to processing queue
        cv2.imshow("AI Assistant", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to process frames (detect objects, colors, faces)
def process_frames(selected_language):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Detect objects
        detected_objects = object_detection.detect_objects(frame)

        # Detect colors of objects
        object_colors = {}
        for obj in detected_objects:
            obj_name, bbox = obj
            x, y, w, h = bbox
            object_color = colour_detection.detect_color(frame, x, y, w, h)
            object_colors[obj_name] = object_color

        # Face Analysis
        face_info = face_analysis.analyze_faces(frame)

        # Construct Description
        description = "I see "
        if detected_objects:
            object_descriptions = []
            for obj_name, bbox in detected_objects:
                color = object_colors.get(obj_name, "unknown color")
                object_descriptions.append(f"a {color} {obj_name}")
            description += ", ".join(object_descriptions) + ". "

        if face_info:
            description += face_info

        if description == "I see ":
            description = "No objects or faces detected."

        print("Generated Description (English):", description)

        # Send to translation & speech thread
        desc_queue.put((description, selected_language))

# Function to translate and speak the description
def translate_and_speak():
    while True:
        description, selected_language = desc_queue.get()
        if description is None:
            break

        # Translate text
        translated_text = translation.translate_text(description, selected_language)
        print(f"Translated Description ({selected_language}):", translated_text)

        # Convert text to speech
        text2speech.speak(translated_text, selected_language)

# Main function to initialize everything
def main():
    print("Starting AI-Powered Blind Assistant...")

    # Get user-selected language through voice
    selected_language = translation.get_language_from_voice()
    print(f"Selected Language: {selected_language}")

    # Start frame capture in a separate thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, args=(selected_language,), daemon=True)
    speech_thread = threading.Thread(target=translate_and_speak, daemon=True)

    capture_thread.start()
    process_thread.start()
    speech_thread.start()

    capture_thread.join()
    process_thread.join()
    speech_thread.join()

if __name__ == "__main__":
    main()
