import pyttsx3
import time

def test_tts():
    print("==== Text-to-Speech Test ====")
    
    try:
        print("Initializing speech engine...")
        engine = pyttsx3.init()
        
        print("Setting speech properties...")
        engine.setProperty('rate', 125)
        engine.setProperty('volume', 1.0)
        
        print("Testing speech output:")
        
        # Test message
        message = "keep speaking and that dick is going into ahmad hassan from brooklyn"
        print(f"Speaking: '{message}'")
        
        engine.say(message)
        engine.runAndWait()
        
        print("\nTTS test completed successfully!")
        
    except Exception as e:
        print(f"Error during TTS test: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure espeak is installed: sudo apt install espeak espeak-ng")
        print("2. Check audio output with test_audio.py")
        print("3. Ensure you're in the correct Python environment")

if __name__ == "__main__":
    test_tts()