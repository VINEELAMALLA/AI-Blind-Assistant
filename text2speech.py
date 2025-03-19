import pyttsx3

def speak_detected_objects(detected_objects):
    """
    Converts detected objects into speech.
    
    :param detected_objects: List of detected objects (e.g., ["person", "car", "dog"])
    """
    engine = pyttsx3.init()
    
    # Set voice properties
    engine.setProperty("rate", 150)  # Speed of speech
    engine.setProperty("volume", 1.0)  # Volume (0.0 to 1.0)

    # Get available voices (to change voice based on preference)
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[0].id)  # Change index for different voices

    if not detected_objects:
        speech_text = "No objects detected."
    else:
        # Convert list to a readable sentence
        speech_text = "I detected " + ", ".join(detected_objects) + " in front of you."

    print("Speaking:", speech_text)
    engine.say(speech_text)
    engine.runAndWait()  # Wait for speech to complete

# Example usage: Objects detected by your AI model
detected_objects = ["person", "car", "bicycle"]
speak_detected_objects(detected_objects)
