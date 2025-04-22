import os
import time
import serial
import RPi.GPIO as GPIO
import numpy as np
import cv2
import subprocess
import shutil
from paths import YOLO_MODEL_PATH, IMAGE_DIR, RESULTS_DIR, LLM_API_URL
from utils.camera_helper import setup_camera, capture_image
from utils.detection import detect_and_classify_holds
from utils.grid_mapping import create_route_visualization
from utils.llm_client import generate_route_description

# Button GPIO pin configurations
CYCLE_BUTTON_PIN = 17  # GPIO pin for cycling through options
SELECT_BUTTON_PIN = 27  # GPIO pin for selecting options

# Servo GPIO pin configuration
SERVO_PIN = 18  # GPIO pin for servo control

# Arduino serial connection
ARDUINO_PORT = '/dev/ttyUSB0'  # Adjust as necessary for your setup
ARDUINO_BAUD = 115200

GRID_WIDTH = 12
GRID_HEIGHT = 12
GRID_SPACING = 15  # mm

time.sleep(2)  # Allow time for Arduino to reset

# TTS engine wrapper class
class DualTTS:
    """Dual TTS engine that uses Pico for quick responses and Google for long text."""
    def __init__(self):
        self.has_pico = self._check_pico()
        self.has_gtts = self._check_gtts()
        
        if not (self.has_pico or self.has_gtts):
            print("WARNING: No TTS engines available. Install either SVOX Pico or gTTS.")
            
        # Initialize pygame for audio playback
        import pygame
        pygame.mixer.init()
        print("TTS initialized with:")
        print(f"- SVOX Pico: {'Available' if self.has_pico else 'Not available'}")
        print(f"- Google TTS: {'Available' if self.has_gtts else 'Not available'}")
            
    def _check_pico(self):
        """Check if SVOX Pico is available."""
        return shutil.which('pico2wave') is not None
    
    def _check_gtts(self):
        """Check if Google TTS is available."""
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False
            
    def speak(self, text, use_google=False):
        """Speak text using the appropriate TTS engine."""
        print(f"TTS: {text}")
        
        if use_google and self.has_gtts:
            self._speak_google(text)
        elif self.has_pico:
            self._speak_pico(text)
        else:
            # Fallback to print only
            print(f"No TTS available. Would say: {text}")
            
    def _speak_pico(self, text):
        """Use SVOX Pico for speech."""
        import pygame
        try:
            wavfile = "/tmp/speech.wav"
            # SVOX Pico has character limits, so split text if needed
            max_chars = 2000
            if len(text) > max_chars:
                chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
                for chunk in chunks:
                    subprocess.run(['pico2wave', '-w', wavfile, chunk], stderr=subprocess.DEVNULL)
                    pygame.mixer.music.load(wavfile)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
            else:
                subprocess.run(['pico2wave', '-w', wavfile, text], stderr=subprocess.DEVNULL)
                pygame.mixer.music.load(wavfile)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"SVOX Pico TTS error: {e}")
    
    def _speak_google(self, text):
        """Use Google TTS for speech with male voice and faster speed."""
        import pygame
        try:
            from gtts import gTTS
            audio_file = "/tmp/speech.mp3"
            
            # Google TTS has limits, so split text if needed
            max_chars = 5000
            if len(text) > max_chars:
                chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
                for chunk in chunks:
                    # Use UK server which typically has a male voice
                    tts = gTTS(text=chunk, lang='en', tld='co.uk', slow=False)
                    tts.save(audio_file)
                    
                    # Load and play audio at faster speed using a trick with pygame
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    
                    # Adjust playback speed (requires separate implementation)
                    self._speed_up_playback(1.3)  # Play at 1.3x speed
                    
            else:
                # Use UK server which typically has a male voice
                tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
                tts.save(audio_file)
                
                # Load and play audio at faster speed
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Adjust playback speed
                self._speed_up_playback(1.3)  # Play at 1.3x speed
                
        except Exception as e:
            print(f"Google TTS error: {e}")
            # Fallback to Pico
            if self.has_pico:
                print("Falling back to SVOX Pico...")
                self._speak_pico(text)
                
    def _speed_up_playback(self, speed_factor):
        """Wait for audio to complete but at an accelerated speed."""
        import pygame
        import time
        
        # Get the length of the audio
        length = pygame.mixer.music.get_pos()
        
        # Calculate wait time based on speed factor
        wait_time = 0.01 / speed_factor
        
        # Wait for playback to complete with adjusted timing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10 * speed_factor)  # Increase tick rate
            time.sleep(wait_time)  # Sleep less time per iteration

# Initialize text-to-speech engine
def init_speech():
    return DualTTS()

# Speak text and wait for completion
def speak(tts, text, use_google=False):
    """Speak text using TTS engine."""
    tts.speak(text, use_google)

# GPIO setup
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(CYCLE_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SELECT_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Add servo control
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    global servo_pwm
    servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz for servos
    servo_pwm.start(2.5)  # Start in retracted position
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)  # Stop PWM to prevent jitter

# Wait for button press with debounce
def wait_for_button_press(pin):
    while GPIO.input(pin) == GPIO.HIGH:  # Wait for button press
        time.sleep(0.05)
    time.sleep(0.2)  # Debounce delay
    while GPIO.input(pin) == GPIO.LOW:  # Wait for button release
        time.sleep(0.05)
    time.sleep(0.2)  # Debounce delay

# Cycle through options with audio feedback - improved with state tracking
def select_from_options(tts, options, prompt):
    speak(tts, prompt) # Use Pico for UI interactions
    current_index = 0
    
    # Announce first option
    speak(tts, f"Option {str(options[current_index])}")
    
    # Button state tracking to avoid multiple triggers per press
    last_cycle_state = False  # For PUD_UP: False means not pressed
    last_select_state = False # For PUD_UP: False means not pressed
    
    while True:
        # With PUD_UP, we need to invert the logic (LOW/False = pressed)
        cycle_pressed = not GPIO.input(CYCLE_BUTTON_PIN)  # LOW (0) means pressed with PUD_UP
        select_pressed = not GPIO.input(SELECT_BUTTON_PIN) # LOW (0) means pressed with PUD_UP
        
        # Detect new press of cycle button (transition from not pressed to pressed)
        if cycle_pressed and not last_cycle_state:
            current_index = (current_index + 1) % len(options)
            speak(tts, f"Option {str(options[current_index])}")
        
        # Detect new press of select button (transition from not pressed to pressed)
        if select_pressed and not last_select_state:
            speak(tts, f"Selected {str(options[current_index])}")
            return options[current_index]
        
        # Update last states
        last_cycle_state = cycle_pressed
        last_select_state = select_pressed
        
        time.sleep(0.1)  # Small delay for debouncing and CPU usage

# Convert grid to GCODE
def grid_to_gcode(grid, x_offset=0, y_offset=0):
    """
    Generate G-code for 2D plotter with servo actuator.
    
    Args:
        grid: The grid representation of the climbing wall
        x_offset: Offset from the plotter's left edge in mm
        y_offset: Offset from the plotter's bottom edge in mm
    """
    gcode = [
        "G21 ; Use millimeters",
        "G90 ; Absolute positioning",
        f"G0 X{-x_offset} Y{-y_offset} F3000 ; Move to custom home position"
    ]

    # Get the actual grid dimensions from the array
    grid_height, grid_width = grid.shape
    grid = np.rot90(grid, k=3)

    
    for y in range(grid_height):
        for x in range(grid_width-1):
            # Read the value from the grid - make sure we're not exceeding array bounds
            val = grid[y, x]
            if val in [1, 2]:
                # Calculate position with negated coordinates to match inverted plotter direction
                pos_x = -(x * GRID_SPACING + x_offset)
                pos_y = -(y * GRID_SPACING + y_offset)  # Invert Y to match physical layout

                # Move to position
                gcode.append(f"G0 X{pos_x} Y{pos_y} F3000 ; Move to ({x},{grid_height-1-y})")
                
                # Activate actuator (servo down) - increased to S1000 for more travel
                gcode.append("M3 S1000 ; Activate actuator (servo up)")
                gcode.append("G4 P0.5 ; Hold for 0.5s")
                
                # Deactivate actuator (servo up)
                gcode.append("M5 ; Deactivate actuator (servo down)")
                gcode.append("G4 P1 ; Wait 1s before next move")

    # Return to custom home position with negated coordinates
    gcode.append(f"G0 X{-x_offset} Y{-y_offset} F3000 ; Return to custom home position")
    return "\n".join(gcode)

# Send GCODE to Arduino
def send_gcode_to_arduino(gcode, tts):
    try:
        # Try to detect Arduino port automatically
        common_ports = ['/dev/ttyUSB0', '/dev/ttyACM0']
        
        # Try to detect Arduino port
        detected_port = None
        for port in common_ports:
            try:
                with serial.Serial(port, 115200, timeout=1) as ser:
                    detected_port = port
                    print(f"Arduino detected on {port}")
                    break
            except:
                pass
        
        if detected_port:
            arduino_port = detected_port
        else:
            arduino_port = ARDUINO_PORT

        ser = serial.Serial(arduino_port, ARDUINO_BAUD, timeout=10)
        print("→ Flushing and waking Arduino...")
        ser.write(b"\r\n\r\n")
        time.sleep(2)
        ser.flushInput()

        for line in gcode.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Send non-servo commands to Arduino normally
            if not line.startswith("M3") and not line.startswith("M5"):
                print(f"Sending: {line}")
                ser.write((line + "\n").encode())
                
                while True:
                    response = ser.readline().decode().strip().lower()
                    if response == "ok":
                        break
                    elif response:
                        print(f"  ↳ Arduino: {response}")
                
                # If this was a movement command, ensure motion is complete
                if line.startswith("G0") or line.startswith("G1"):
                    print("Waiting for movement to complete...")
                    # Wait until machine is idle
                    is_moving = True
                    while is_moving:
                        # Send status request
                        ser.write(b"?")
                        status_response = ser.readline().decode().strip()
                        # Check if status contains "Idle" indicating movement complete
                        if "idle" in status_response.lower():
                            is_moving = False
                            print("Movement complete.")
                        else:
                            # Brief pause to avoid flooding controller
                            time.sleep(0.1)
                
            # Handle servo control with Pi GPIO
            elif line.startswith("M3 S"):
                print("Pi controlling servo: EXTEND")
    
                # First push
                servo_pwm.ChangeDutyCycle(12.0)
                time.sleep(0.5)
                
                # Brief slight retraction
                servo_pwm.ChangeDutyCycle(11.0)
                time.sleep(0.2)
                
                # Second push - stronger
                servo_pwm.ChangeDutyCycle(12.5)  # Push a bit further
                time.sleep(0.5)
                    
                # Send placeholder command to Arduino for synchronization
                ser.write(b"G4 P0\n")
                while True:
                    response = ser.readline().decode().strip().lower()
                    if response == "ok":
                        break
                    elif response:
                        print(f"  ↳ Arduino: {response}")
                    
            elif line.startswith("M5"):
                print("Pi controlling servo: RETRACT")
                servo_pwm.ChangeDutyCycle(2.5)  # Retract position
                time.sleep(0.5)  # Increased from 0.1 to 0.5 for reliable retraction
                servo_pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter
                
                # Send placeholder command to Arduino for synchronization
                ser.write(b"G4 P0\n")
                while True:
                    response = ser.readline().decode().strip().lower()
                    if response == "ok":
                        break
                    elif response:
                        print(f"  ↳ Arduino: {response}")
        
        return True
    except Exception as e:
        print(f"Error communicating with Arduino: {e}")
        return False

def display_text_grid(grid_map: np.ndarray):
    """
    Display a text-based representation of the climbing route grid in the terminal.
    
    Args:
        grid_map: 12x12 numpy array representing the route
    """
    print("\nRoute Grid Map (Text Representation):")
    print("  " + "".join([f"{i:2d}" for i in range(grid_map.shape[1])]))
    print("  " + "-" * (grid_map.shape[1] * 2))
    
    for i, row in enumerate(grid_map):
        line = f"{i:2d}|"
        for cell in row:
            if cell == 0:
                line += ". "  # Empty space
            elif cell == 1:
                line += "o "  # Small hold
            elif cell == 2:
                line += "O "  # Large hold
        print(line)
    
    print("\nLegend: '.' = Empty, 'o' = Small hold, 'O' = Large hold")

# Main function
def main():
    # Initialize components
    tts = init_speech()
    setup_gpio()
    
    try:
        # Welcome message
        speak(tts, "Welcome to the climbing route analyzer for visually impaired users.")
        speak(tts, "This system will scan the climbing wall and create a tactile map.")
        
        # Ask if user wants to use camera or image from directory
        use_camera = select_from_options(tts, ["No", "Yes"], 
                                         "Do you want to use the camera to capture a new image?") == "Yes"
        
        # Camera or file processing
        if use_camera:
            speak(tts, "Setting up camera. Please point it at the climbing wall.")
            # Default to Pi camera for simplicity
            camera = setup_camera(use_picamera=True)
            
            if camera is None:
                speak(tts, "Failed to initialize camera. Exiting.")
                return
                
            # Ensure results directory exists
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
                
            speak(tts, "Press the select button to capture the image.")
            wait_for_button_press(SELECT_BUTTON_PIN)
            
            speak(tts, "Capturing image...")
            image_path = capture_image(camera, save_path=os.path.join(RESULTS_DIR, "captured_image.jpg"))
            
            if hasattr(camera, 'release'):
                camera.release()
                
            if image_path is None:
                speak(tts, "Failed to capture image. Exiting.")
                return
                
            speak(tts, "Image captured successfully.")
        else:
            # Use an image from the directory
            image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                speak(tts, "No images found in the specified directory. Exiting.")
                return
            
            image_path = os.path.join(IMAGE_DIR, image_files[0])
            speak(tts, f"Using the image file {os.path.basename(image_path)}")
            
        # Select color for detection
        color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink"]
        target_color = select_from_options(tts, color_options, 
                                          "Please select the color of holds to detect.")
        
        # Use auto-sensitivity based on image brightness
        speak(tts, "Calculating optimal sensitivity based on image brightness.")
        import cv2
        temp_img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV)
        brightness = hsv_img[:, :, 2].mean()
        sensitivity = int(60 - (brightness / 255) * 40)
        sensitivity = max(10, min(60, sensitivity))
        min_area = 30  # Default minimum area
        
        speak(tts, f"Processing image to detect {target_color} holds.")
        
        # Detect holds and map to grid
        holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
            image_path,
            target_color=target_color,
            sensitivity=sensitivity,
            min_area=min_area,
            yolo_model_path=YOLO_MODEL_PATH
        )
        
        if not holds_info:
            speak(tts, "No holds detected. Please try again with different settings.")
            return
            
        num_holds = len(holds_info)
        speak(tts, f"Detection complete. Found {num_holds} {target_color} holds on the wall.")

        display_text_grid(grid_map)
                
        # Ask if user wants to feel the tactile representation
        use_tactile = select_from_options(tts, ["Yes", "No"], 
                                         "Would you like to create a tactile representation of the route?") == "Yes"
        
        if use_tactile:
            speak(tts, "Generating GCODE for the tactile display. This will take a moment.")
            
            # Let user configure the home position
            #speak(tts, "Setting up the plotter position.")
            # Default offsets, adjust based on your setup
            x_offset = 0  # 20mm from the left edge
            y_offset = 0  # 20mm from the bottom edge
            
            # Convert grid to GCODE with custom offset
            gcode = grid_to_gcode(grid_map, x_offset=x_offset, y_offset=y_offset)
            
            # Save GCODE to file for reference
            gcode_path = os.path.join(RESULTS_DIR, "route_tactile.gcode")
            with open(gcode_path, 'w') as f:
                f.write(gcode)
            
            # Send to Arduino
            speak(tts, "Sending route to tactile display. This may take a few minutes.")
            if send_gcode_to_arduino(gcode, tts):
                speak(tts, "Tactile representation complete. You can now explore the route.")
            else:
                speak(tts, "Failed to send tactile display. Check connections and try again.")
        
        speak(tts, "Generating route description. Please wait...")
        description = generate_route_description(grid_map, use_local_llm=False, api_url=LLM_API_URL)
        
        # Speak the description using Google TTS for higher quality
        speak(tts, "Here is the description of the climbing route:", use_google=False)
        speak(tts, "Press the cycle button to skip at any time.", use_google=False)
        # Use Google TTS for the route description (better quality)
        speak(tts, description, use_google=True)

        while True:
            repeat = select_from_options(tts, ["Yes", "No"], 
                                      "Would you like to hear the description again?")
            if repeat == "Yes":
                 # Continue using Google TTS for the route description
                 speak(tts, description, use_google=True)
            else:
                break

        # Completion
        speak(tts, "Analysis complete. Thank you for using the climbing route analyzer.")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        speak(tts, f"An error occurred: {str(e)}")
    finally:
        # Properly stop PWM before cleanup
        if 'servo_pwm' in globals():
            try:
                servo_pwm.ChangeDutyCycle(0)
                time.sleep(0.1)
                servo_pwm.stop()
                time.sleep(0.1)
            except Exception as e:
                print(f"Warning: Error stopping servo PWM: {e}")
        
        # Clean up with error handling
        try:
            GPIO.cleanup()
            print("GPIO cleanup completed successfully")
        except Exception as e:
            print(f"Warning: Error during GPIO cleanup: {e}")

if __name__ == "__main__":
    main()
