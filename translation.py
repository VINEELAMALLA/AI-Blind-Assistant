import speech_recognition as sr
from gtts import gTTS
import os
from googletrans import Translator

# Language Mapping
LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "spanish": "es",
    "french": "fr",
    "telugu": "te"
}

# Initialize Translator
translator = Translator()

def recognize_language():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak the language you want (English, Hindi, Spanish, French, Telugu)...")
        recognizer.adjust_for_ambient_noise(source)
        
        try:
            audio = recognizer.listen(source)
            spoken_text = recognizer.recognize_google(audio).lower()
            print(f"🗣️ You said: {spoken_text}")
            
            if spoken_text in LANGUAGE_MAP:
                return LANGUAGE_MAP[spoken_text]
            else:
                print("❌ Language not recognized. Defaulting to English.")
                return "en"
        except sr.UnknownValueError:
            print("❌ Could not understand the audio. Defaulting to English.")
            return "en"
        except sr.RequestError:
            print("❌ Speech recognition service error. Defaulting to English.")
            return "en"

def text_to_speech(text, language):
    # Translate text if needed
    if language != "en":
        translated_text = translator.translate(text, dest=language).text
        print(f"🔄 Translated: {translated_text}")
    else:
        translated_text = text
    
    tts = gTTS(text=translated_text, lang=language)
    tts.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "mpg321 output.mp3")

# 🔹 Get the user's language choice via voice
selected_language = recognize_language()

# 🔹 Example: Describe a detected object
object_description = "A person is wearing a red shirt."
text_to_speech(object_description, selected_language)
