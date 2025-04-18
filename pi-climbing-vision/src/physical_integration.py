import os
import time
import pyttsx3
import serial
import RPi.GPIO as GPIO
from paths import YOLO_MODEL_PATH, IMAGE_DIR, RESULTS_DIR, LLM_API_URL
from utils.camera_helper import setup_camera, capture_image
from utils.detection import detect_and_classify_holds
from utils.grid_mapping import create_route_visualization
from utils.llm_client import generate_route_description

# Button GPIO pin configurations
CYCLE_BUTTON_PIN = 17  # GPIO pin for cycling through options
SELECT_BUTTON_PIN = 27  # GPIO pin for selecting options

# Arduino serial connection
ARDUINO_PORT = '/dev/ttyUSB0'  # Adjust as necessary for your setup
ARDUINO_BAUD = 115200
time.sleep(2)  # Allow time for Arduino to reset

# Initialize text-to-speech engine
def init_speech():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    return engine

# GPIO setup
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    # You can use either PUD_UP or PUD_DOWN depending on your hardware setup
    # With PUD_UP: Button press = LOW, Released = HIGH
    # With PUD_DOWN: Button press = HIGH, Released = LOW
    GPIO.setup(CYCLE_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SELECT_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Wait for button press with debounce
def wait_for_button_press(pin):
    while GPIO.input(pin) == GPIO.HIGH:  # Wait for button press
        time.sleep(0.05)
    time.sleep(0.2)  # Debounce delay
    while GPIO.input(pin) == GPIO.LOW:  # Wait for button release
        time.sleep(0.05)
    time.sleep(0.2)  # Debounce delay

# Speak text and wait for completion
def speak(engine, text):
    print(text)  # For debugging
    engine.say(text)
    engine.runAndWait()

# Cycle through options with audio feedback - improved with state tracking
def select_from_options(engine, options, prompt):
    speak(engine, prompt)
    current_index = 0
    
    # Announce first option
    speak(engine, f"Option: {options[current_index]}")
    
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
            speak(engine, f"Option: {options[current_index]}")
        
        # Detect new press of select button (transition from not pressed to pressed)
        if select_pressed and not last_select_state:
            speak(engine, f"Selected: {options[current_index]}")
            return options[current_index]
        
        # Update last states
        last_cycle_state = cycle_pressed
        last_select_state = select_pressed
        
        time.sleep(0.1)  # Small delay for debouncing and CPU usage

# Convert grid to GCODE

def grid_to_gcode(grid_map, bed_size_x=200, bed_size_y=200):
    """Convert 12x12 grid to GCODE for tactile representation"""
    gcode = []
    
    # Start with homing
    gcode.append("G28 ; Home all axes")
    
    # Calculate cell size
    cell_width = bed_size_x / 12
    cell_height = bed_size_y / 12
    
    # Add header
    gcode.append("M107 ; Turn off fan")
    gcode.append("G90 ; Absolute positioning")
    gcode.append("M83 ; Relative extruder mode")
    
    # Move through each cell in the grid
    for y in range(12):
        for x in range(12):
            # Calculate center position of this cell
            pos_x = (x + 0.5) * cell_width
            pos_y = (y + 0.5) * cell_height
            
            # Move to position
            gcode.append(f"G0 X{pos_x:.2f} Y{pos_y:.2f} F3000 ; Move to cell ({x},{y})")
            
            # If there's a hold at this position, activate the actuator
            if grid_map[11-y][x] > 0:  # Note: Grid is inverted, bottom is y=11
                # Different heights based on hold size (1=small, 2=large)
                height = 5 if grid_map[11-y][x] == 1 else 10
                gcode.append(f"M280 P0 S{90+height} ; Raise pin to {height}mm")
                gcode.append("G4 P500 ; Wait 500ms")
                gcode.append("M280 P0 S90 ; Lower pin")
                gcode.append("G4 P200 ; Wait 200ms")
    
    # Return to origin when done
    gcode.append("G0 X0 Y0 F3000 ; Return to origin")
    
    return "\n".join(gcode)
'''
def grid_to_gcode(grid_map, grid_width=12, grid_height=12):
    lines = []

    for y in range(grid_height):
        for x in range(grid_width):
            # Remember: grid is bottom-up (0 at bottom), so invert y
            if grid_map[grid_height - 1 - y][x] > 0:
                lines.append(f"{x},{y}")

    return "\n".join(lines)
'''
    
# Send GCODE to Arduino
def send_gcode_to_arduino(gcode, engine):
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
            print("No Arduino automatically detected.")
            port = input(f"Enter Arduino port (default: {ARDUINO_PORT}): ").strip()
            if port:
                arduino_port = port
            else:
                arduino_port = ARDUINO_PORT

        with serial.Serial(arduino_port, ARDUINO_BAUD, timeout=10) as ser:
            speak(engine, "Connecting to tactile display...")
            time.sleep(2)  # Allow time for Arduino to initialize
            
            # Send GCODE line by line
            for line in gcode.split("\n"):
                print(f"Sending: {line}")
                ser.write((line + "\n").encode())
                time.sleep(0.1)  # Small delay to allow Arduino to process
            
            return True
    except Exception as e:
        print(f"Error communicating with Arduino: {e}")
        return False

# Main function
def main():
    # Initialize components
    engine = init_speech()
    setup_gpio()
    
    try:
        # Welcome message
        speak(engine, "Welcome to the climbing route analyzer for visually impaired users.")
        speak(engine, "This system will scan the climbing wall and create a tactile map.")
        
        # Ask if user wants to use camera or image from directory
        use_camera = select_from_options(engine, ["Yes", "No"], 
                                         "Do you want to use the camera to capture a new image?") == "Yes"
        
        # Camera or file processing
        if use_camera:
            speak(engine, "Setting up camera. Please point it at the climbing wall.")
            # Default to Pi camera for simplicity
            camera = setup_camera(use_picamera=True)
            
            if camera is None:
                speak(engine, "Failed to initialize camera. Exiting.")
                return
                
            # Ensure results directory exists
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
                
            speak(engine, "Press the select button to capture the image.")
            wait_for_button_press(SELECT_BUTTON_PIN)
            
            speak(engine, "Capturing image...")
            image_path = capture_image(camera, save_path=os.path.join(RESULTS_DIR, "captured_image.jpg"))
            
            if hasattr(camera, 'release'):
                camera.release()
                
            if image_path is None:
                speak(engine, "Failed to capture image. Exiting.")
                return
                
            speak(engine, "Image captured successfully.")
        else:
            # Use an image from the directory
            image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                speak(engine, "No images found in the specified directory. Exiting.")
                return
            
            image_path = os.path.join(IMAGE_DIR, image_files[0])
            speak(engine, f"Using the image file {os.path.basename(image_path)}")
            
        # Select color for detection
        color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink"]
        target_color = select_from_options(engine, color_options, 
                                          "Please select the color of holds to detect.")
        
        # Use auto-sensitivity based on image brightness
        speak(engine, "Calculating optimal sensitivity based on image brightness.")
        import cv2
        temp_img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV)
        brightness = hsv_img[:, :, 2].mean()
        sensitivity = int(60 - (brightness / 255) * 40)
        sensitivity = max(10, min(60, sensitivity))
        min_area = 200  # Default minimum area
        
        speak(engine, f"Processing image to detect {target_color} holds.")
        
        # Detect holds and map to grid
        holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
            image_path,
            target_color=target_color,
            sensitivity=sensitivity,
            min_area=min_area,
            yolo_model_path=YOLO_MODEL_PATH
        )
        
        if not holds_info:
            speak(engine, "No holds detected. Please try again with different settings.")
            return
            
        num_holds = len(holds_info)
        speak(engine, f"Detection complete. Found {num_holds} {target_color} holds on the wall.")
        
        # Generate route description
        speak(engine, "Generating route description. Please wait...")
        description = generate_route_description(grid_map, use_local_llm=False, api_url=LLM_API_URL)
        
        # Speak the description
        speak(engine, "Here is the description of the climbing route:")
        speak(engine, description)
        
        # Ask if user wants to feel the tactile representation
        use_tactile = select_from_options(engine, ["Yes", "No"], 
                                         "Would you like to create a tactile representation of the route?") == "Yes"
        
        if use_tactile:
            speak(engine, "Generating GCODE for the tactile display. This will take a moment.")
            
            # Convert grid to GCODE
            gcode = grid_to_gcode(grid_map)
            
            # Save GCODE to file for reference
            gcode_path = os.path.join(RESULTS_DIR, "route_tactile.gcode")
            with open(gcode_path, 'w') as f:
                f.write(gcode)
            
            # Send to Arduino
            speak(engine, "Sending route to tactile display. This may take a few minutes.")
            if send_gcode_to_arduino(gcode, engine):
                speak(engine, "Tactile representation complete. You can now explore the route.")
            else:
                speak(engine, "Failed to send route to tactile display. Check connections and try again.")
        
        # Completion
        speak(engine, "Analysis complete. Thank you for using the climbing route analyzer.")
        
    except Exception as e:
        speak(engine, f"An error occurred: {str(e)}")
    finally:
        # Clean up
        GPIO.cleanup()

if __name__ == "__main__":
    main()
