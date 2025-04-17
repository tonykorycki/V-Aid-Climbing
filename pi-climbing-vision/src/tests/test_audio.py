import os
import time

def test_audio():
    print("==== Audio Test ====")
    print("Testing basic audio output with tone...")
    
    # Test with speaker-test
    os.system("speaker-test -t sine -f 440 -c 2 -s 2")
    
    print("\nIf you heard audio tones, the basic audio system is working!")
    print("For Text-to-Speech testing, run test_tts.py")

if __name__ == "__main__":
    test_audio()