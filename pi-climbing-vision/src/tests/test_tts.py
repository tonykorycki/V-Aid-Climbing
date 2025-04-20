import pyttsx3
import time
import os
import subprocess
import shutil

def test_pyttsx3():
    """Test the default pyttsx3 engine"""
    print("\n=== Testing pyttsx3 (default) ===")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 125)
        engine.setProperty('volume', 1.0)
        message = "This is pyttsx3 text to speech. Is this clear enough?"
        print(f"Speaking: '{message}'")
        engine.say(message)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"Error with pyttsx3: {e}")
        return False

def test_pico():
    """Test SVOX Pico TTS"""
    print("\n=== Testing SVOX Pico TTS ===")
    if not shutil.which('pico2wave'):
        print("SVOX Pico not installed. Run 'sudo apt-get install -y libttspico-utils'")
        return False
        
    try:
        message = "This is SVOX Pico text to speech. Is this clear enough?"
        print(f"Speaking: '{message}'")
        wavfile = "/tmp/test.wav"
        subprocess.run(['pico2wave', '-w', wavfile, message])
        subprocess.run(['aplay', wavfile])
        return True
    except Exception as e:
        print(f"Error with SVOX Pico: {e}")
        return False

def test_festival():
    """Test Festival TTS"""
    print("\n=== Testing Festival TTS ===")
    if not shutil.which('text2wave'):
        print("Festival not installed. Run 'sudo apt-get install -y festival'")
        return False
        
    try:
        message = "This is Festival text to speech. Is this clear enough?"
        print(f"Speaking: '{message}'")
        with open('/tmp/test_festival.txt', 'w') as f:
            f.write(message)
        subprocess.run(['text2wave', '/tmp/test_festival.txt', '-o', '/tmp/test_festival.wav'])
        subprocess.run(['aplay', '/tmp/test_festival.wav'])
        return True
    except Exception as e:
        print(f"Error with Festival: {e}")
        return False

def test_mbrola():
    """Test MBROLA enhanced eSpeak"""
    print("\n=== Testing MBROLA enhanced eSpeak ===")
    try:
        check = subprocess.run(['espeak-ng', '--voices=mb'], 
                             stdout=subprocess.PIPE).stdout.decode()
        if 'mb-us1' not in check:
            print("MBROLA voices not installed. Run 'sudo apt-get install -y mbrola mbrola-us1'")
            return False
            
        message = "This is MBROLA enhanced text to speech. Is this clear enough?"
        print(f"Speaking: '{message}'")
        subprocess.run(['espeak-ng', '-v', 'mb-us1', message])
        return True
    except Exception as e:
        print(f"Error with MBROLA: {e}")
        return False

def test_gtts():
    """Test Google TTS (requires internet)"""
    print("\n=== Testing Google TTS (requires internet) ===")
    try:
        from gtts import gTTS
        import os
        
        message = "This is Google text to speech. Is this clear enough?"
        print(f"Speaking: '{message}'")
        
        if not shutil.which('mpg321'):
            print("mpg321 not installed. Run 'sudo apt-get install -y mpg321'")
            return False
            
        tts = gTTS(text=message, lang='en')
        tts.save("/tmp/speech.mp3")
        os.system("mpg321 /tmp/speech.mp3")
        return True
    except ImportError:
        print("gTTS not installed. Run 'pip install gtts'")
        return False
    except Exception as e:
        print(f"Error with Google TTS: {e}")
        return False

def test_tts():
    print("==== Advanced Text-to-Speech Test ====")
    print("Testing multiple TTS engines to find the clearest option.")
    
    results = []
    
    # Test default pyttsx3 (for reference)
    results.append(("pyttsx3", test_pyttsx3()))
    time.sleep(1)
    
    # Test SVOX Pico
    results.append(("SVOX Pico", test_pico()))
    time.sleep(1)
    
    # Test Festival
    results.append(("Festival", test_festival()))
    time.sleep(1)
    
    # Test MBROLA
    results.append(("MBROLA", test_mbrola()))
    time.sleep(1)
    
    # Test Google TTS
    results.append(("Google TTS", test_gtts()))
    
    # Print summary
    print("\n==== TTS Test Results ====")
    working_engines = []
    for engine, success in results:
        status = "✓ WORKING" if success else "✗ FAILED"
        print(f"{engine}: {status}")
        if success:
            working_engines.append(engine)
    
    if working_engines:
        print(f"\n🎤 Working TTS engines: {', '.join(working_engines)}")
        print("Choose the clearest one for your application!")
        print("\nUse instructions in the documentation to change the default TTS engine.")
    else:
        print("\n❌ No TTS engines working properly. Please check installation.")

if __name__ == "__main__":
    test_tts()