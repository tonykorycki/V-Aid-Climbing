import serial
import time
import sys
import RPi.GPIO as GPIO  # Add GPIO import
try:
    import serial.tools.list_ports  # For port detection
except ImportError:
    print("Warning: serial.tools.list_ports not available. Port detection disabled.")

# ===== Default Configurable Constants =====
DEFAULT_GRID_WIDTH = 11
DEFAULT_GRID_HEIGHT = 12
DEFAULT_GRID_SPACING = 15  # mm between each pin
DEFAULT_OFFSET_X = 0       # mm from X=0 to first column
DEFAULT_OFFSET_Y = 0      # mm from Y=0 to first row
DEFAULT_FEEDRATE = 3000    # Movement speed
DEFAULT_SERVO_PUSH_DELAY = 0.5  # seconds to hold actuator
DEFAULT_SERVO_RETRACT_DELAY = 0.5  # seconds after actuator retract
DEFAULT_ARDUINO_BAUD = 115200
SERVO_PIN = 18  # GPIO pin for servo control

# Try to auto-detect port or use default
try:
    import platform
    if platform.system() == "Windows":
        DEFAULT_ARDUINO_PORT = 'COM4'
    else:
        DEFAULT_ARDUINO_PORT = '/dev/ttyUSB0'
except:
    DEFAULT_ARDUINO_PORT = '/dev/ttyUSB0'  # Fallback

# Initialize servo PWM control
def setup_servo():
    """Initialize the servo for direct control from Pi GPIO"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    global servo_pwm
    servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
    servo_pwm.start(2.5)  # Initialize in retracted position
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)  # Stop PWM signal to prevent jitter
    print("✓ Servo control initialized on GPIO pin", SERVO_PIN)

# ===== Arduino Detection =====
def detect_arduino_ports():
    """Scan for available serial ports and identify potential Arduino connections."""
    try:
        arduino_ports = []
        print("\n==== Scanning for Arduino devices ====")
        
        # Get list of all available ports
        available_ports = list(serial.tools.list_ports.comports())
        
        if not available_ports:
            print("❌ No serial ports found. Check if device is connected.")
            return []
            
        print(f"Found {len(available_ports)} serial port(s):")
        
        # Check each port for Arduino identifiers
        for port in available_ports:
            port_info = f"- {port.device}"
            
            # Add description if available
            if port.description:
                port_info += f" ({port.description})"
            
            # Arduino often has these strings in manufacturer or description
            is_arduino = any(identifier.lower() in (port.manufacturer or "").lower() + 
                            (port.description or "").lower() for identifier in 
                            ["arduino", "ch340", "ftdi", "wch", "usb serial"])
            
            if is_arduino:
                port_info += " [LIKELY ARDUINO]"
                arduino_ports.append(port.device)
                
            print(port_info)
        
        if arduino_ports:
            print(f"\n✅ Found {len(arduino_ports)} probable Arduino port(s): {', '.join(arduino_ports)}")
        else:
            print("\n⚠️ No obvious Arduino devices found. Try each port manually.")
            # Return all ports if no Arduino ports identified
            arduino_ports = [port.device for port in available_ports]
        
        return arduino_ports
        
    except Exception as e:
        print(f"❌ Error detecting ports: {e}")
        return []

# ===== User Configuration =====
def get_user_config():
    """Get configuration values from user with defaults."""
    print("\n===== PLOTTER CONFIGURATION =====")
    print("Press Enter to use default values or input new ones.\n")
    
    try:
        # Get offsets
        offset_x_input = input(f"X offset in mm (default: {DEFAULT_OFFSET_X}): ").strip()
        offset_x = float(offset_x_input) if offset_x_input else DEFAULT_OFFSET_X
        
        offset_y_input = input(f"Y offset in mm (default: {DEFAULT_OFFSET_Y}): ").strip()
        offset_y = float(offset_y_input) if offset_y_input else DEFAULT_OFFSET_Y
        
        # Get grid spacing
        spacing_input = input(f"Grid spacing in mm (default: {DEFAULT_GRID_SPACING}): ").strip()
        grid_spacing = float(spacing_input) if spacing_input else DEFAULT_GRID_SPACING
        
        # Get Arduino port
        port_input = input(f"Arduino port (default: {DEFAULT_ARDUINO_PORT}): ").strip()
        arduino_port = port_input if port_input else DEFAULT_ARDUINO_PORT
        
        # Use default values for other parameters
        config = {
            "GRID_WIDTH": DEFAULT_GRID_WIDTH,
            "GRID_HEIGHT": DEFAULT_GRID_HEIGHT,
            "GRID_SPACING": grid_spacing,
            "OFFSET_X": offset_x,
            "OFFSET_Y": offset_y,
            "FEEDRATE": DEFAULT_FEEDRATE,
            "SERVO_PUSH_DELAY": DEFAULT_SERVO_PUSH_DELAY,
            "SERVO_RETRACT_DELAY": DEFAULT_SERVO_RETRACT_DELAY,
            "ARDUINO_PORT": arduino_port,
            "ARDUINO_BAUD": DEFAULT_ARDUINO_BAUD
        }
        
        # Show configuration summary
        print("\n===== CONFIGURATION SUMMARY =====")
        print(f"Grid size: {config['GRID_WIDTH']}x{config['GRID_HEIGHT']}")
        print(f"Grid spacing: {config['GRID_SPACING']} mm")
        print(f"X offset: {config['OFFSET_X']} mm")
        print(f"Y offset: {config['OFFSET_Y']} mm")
        print(f"Arduino port: {config['ARDUINO_PORT']}")
        
        return config
    except ValueError:
        print("Invalid input. Using default values.")
        return {
            "GRID_WIDTH": DEFAULT_GRID_WIDTH,
            "GRID_HEIGHT": DEFAULT_GRID_HEIGHT,
            "GRID_SPACING": DEFAULT_GRID_SPACING,
            "OFFSET_X": DEFAULT_OFFSET_X,
            "OFFSET_Y": DEFAULT_OFFSET_Y,
            "FEEDRATE": DEFAULT_FEEDRATE,
            "SERVO_PUSH_DELAY": DEFAULT_SERVO_PUSH_DELAY,
            "SERVO_RETRACT_DELAY": DEFAULT_SERVO_RETRACT_DELAY,
            "ARDUINO_PORT": DEFAULT_ARDUINO_PORT,
            "ARDUINO_BAUD": DEFAULT_ARDUINO_BAUD
        }

# ===== G-code Logic =====
def generate_full_grid_gcode(config):
    """Generate G-code to sequentially push all pins on a grid."""
    gcode = [
        "G21 ; Use millimeters",
        "G90 ; Absolute positioning",
        f"G0 X{-config['OFFSET_X']} Y{-config['OFFSET_Y']} F{config['FEEDRATE']} ; Move to custom home position"
    ]
    
    # This will be separated to allow confirmation
    home_gcode = "\n".join(gcode)
    
    grid_gcode = []
    for row in range(config['GRID_HEIGHT']):
        for col in range(config['GRID_WIDTH']):
            # Negate X and Y coordinates to match inverted plotter direction
            x = -(config['OFFSET_X'] + col * config['GRID_SPACING'])
            y = -(config['OFFSET_Y'] + row * config['GRID_SPACING'])
            grid_gcode.append(f"G0 X{x} Y{y} F{config['FEEDRATE']} ; Move to ({col},{row})")
            grid_gcode.append("M3 S1000 ; Push pin (servo activate max)")
            grid_gcode.append(f"G4 P{config['SERVO_PUSH_DELAY']} ; Wait while pushing")
            grid_gcode.append("M5 ; Retract actuator")
            grid_gcode.append(f"G4 P{config['SERVO_RETRACT_DELAY']} ; Wait after retract")

    grid_gcode.append(f"G0 X{-config['OFFSET_X']} Y{-config['OFFSET_Y']} F{config['FEEDRATE']} ; Return to custom home")
    
    return home_gcode, "\n".join(grid_gcode)

# ===== Serial Communication =====
def send_gcode(ser, gcode, wait_for_confirmation=False):
    """Send G-code to Arduino with optional confirmation."""
    for line in gcode.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        # Send non-servo commands to Arduino normally
        if not line.startswith("M3") and not line.startswith("M5"):
            print(f"Sending: {line}")
            ser.write((line + "\n").encode())

            # Wait for "ok" from Arduino
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
                    # Check if status contains "Idle" or "Idle:" indicating movement complete
                    if "idle" in status_response.lower():
                        is_moving = False
                        print("Movement complete.")
                    else:
                        # Brief pause to avoid flooding controller
                        time.sleep(0.1)
            
        # Handle servo control with Pi GPIO
        elif line.startswith("M3 S"):
            print("Pi controlling servo: EXTEND")
            servo_pwm.ChangeDutyCycle(12.5)
            time.sleep(0.7)
            
            # Brief slight retraction
            servo_pwm.ChangeDutyCycle(8.0)
            time.sleep(0.3)
            
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
            time.sleep(0.5)
            servo_pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter
            
            # Send placeholder command to Arduino for synchronization
            ser.write(b"G4 P0\n")
            while True:
                response = ser.readline().decode().strip().lower()
                if response == "ok":
                    break
                elif response:
                    print(f"  ↳ Arduino: {response}")

# ===== Test Options =====
def move_to_corner_test(config, ser):
    """Test moving to each corner of the grid."""
    print("\n==== Corner Movement Test ====")
    
    corners = [
        (0, 0, "bottom-left"),
        (config['GRID_WIDTH']-1, 0, "bottom-right"),
        (0, config['GRID_HEIGHT']-1, "top-left"),
        (config['GRID_WIDTH']-1, config['GRID_HEIGHT']-1, "top-right")
    ]
    
    gcode = ["G21 ; Use millimeters", "G90 ; Absolute positioning"]
    
    for col, row, name in corners:
        # Negate X and Y coordinates for inverted direction
        x = -(config['OFFSET_X'] + col * config['GRID_SPACING'])
        y = -(config['OFFSET_Y'] + row * config['GRID_SPACING'])
        print(f"\nMoving to {name} corner ({col}, {row})...")
        cmd = f"G0 X{x} Y{y} F{config['FEEDRATE']}"
        ser.write((cmd + "\n").encode())
        
        while True:
            response = ser.readline().decode().strip().lower()
            if response == "ok":
                break
            elif response:
                print(f"  ↳ Arduino: {response}")
        
        input(f">>> Now at {name} corner. Press ENTER to continue <<<")
    
    # Return to home
    home_x = -config['OFFSET_X']
    home_y = -config['OFFSET_Y']
    cmd = f"G0 X{home_x} Y{home_y} F{config['FEEDRATE']}"
    ser.write((cmd + "\n").encode())
    while True:
        response = ser.readline().decode().strip().lower()
        if response == "ok":
            break

def reset_coordinates(ser, config):
    """Reset the coordinate system to match new configuration."""
    print("\n→ Resetting coordinate system...")
    
    # Set absolute positioning
    reset_gcode = "G21 ; Use millimeters\n"
    reset_gcode += "G90 ; Absolute positioning\n"
    
    # Set current position as (0,0)
    reset_gcode += "G92 X0 Y0 ; Reset coordinate system"
    
    # Send the reset commands
    for line in reset_gcode.split("\n"):
        if not line.strip():
            continue
        print(f"Sending: {line}")
        ser.write((line + "\n").encode())
        
        # Wait for acknowledgement
        while True:
            response = ser.readline().decode().strip().lower()
            if response == "ok":
                break
            elif response:
                print(f"  ↳ Arduino: {response}")
    
    print("✓ Coordinate system reset. Ready for new commands.")

def calibrate_home_position(ser, config):
    # Reset coordinates before test
    reset_coordinates(ser, config)
    # Home position calibration
    home_gcode = "G21 ; Use millimeters\nG90 ; Absolute positioning\n"
    home_gcode += f"G0 X{-config['OFFSET_X']} Y{-config['OFFSET_Y']} F{config['FEEDRATE']}"
    send_gcode(ser, home_gcode, wait_for_confirmation=True)
    print("✅ Home position verified.")

def quick_servo_test(ser):
    """Test servo actuator using Pi control."""
    print("\nTesting servo actuator directly from Raspberry Pi...")
    print("Extending servo...")
    servo_pwm.ChangeDutyCycle(12.5)  # Full extension
    time.sleep(1)
    
    print("Retracting servo...")
    servo_pwm.ChangeDutyCycle(2.5)  # Retracted position
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)  # Stop signal
    
    print("✅ Servo actuator test complete.")

# ===== Main Script =====
def full_grid_test():
    """Run the full grid test with user configuration."""
    print("\n==== V-Aid Climbing Plotter Debug Utility ====")
    
    # Initialize servo control
    setup_servo()
    
    # Detect Arduino ports first
    arduino_ports = detect_arduino_ports()
    if arduino_ports:
        # Update default port to first detected Arduino
        global DEFAULT_ARDUINO_PORT
        DEFAULT_ARDUINO_PORT = arduino_ports[0]
    
    while True:
        # Get user configuration
        config = get_user_config()
        
        try:
            print(f"\nConnecting to Arduino on {config['ARDUINO_PORT']}...")
            ser = serial.Serial(config['ARDUINO_PORT'], config['ARDUINO_BAUD'], timeout=5)
            
            # Initialize GRBL firmware
            print("→ Flushing and waking Arduino...")
            ser.write(b"\r\n\r\n")  # Wake/reset GRBL firmware
            time.sleep(2)          # Allow time for reset
            ser.flushInput()       # Clear any startup messages
            
            # Reset coordinates to match new configuration
            reset_coordinates(ser, config)
            
            print("✓ Serial connection established and GRBL initialized.")
            
            while True:
                print("\nSelect test to run:")
                print("1. Full grid test (actuate all positions)")
                print("2. Corner movement test (check grid boundaries)")
                print("3. Calibrate home position")
                print("4. Quick servo test (test actuator only)")
                print("5. Scan for Arduino ports")
                print("6. Update configuration")
                print("7. Exit program")
                choice = input("Enter choice (1-7): ").strip()
                
                if choice == "1":
                    # Reset coordinates before test
                    reset_coordinates(ser, config)
                    # Full grid test with confirmation
                    home_gcode, grid_gcode = generate_full_grid_gcode(config)
                    
                    # First move to home and confirm
                    print("\nMoving to home position for confirmation...")
                    send_gcode(ser, home_gcode, wait_for_confirmation=True)
                    
                    # Then run the full grid test
                    print("\nRunning full grid test...")
                    send_gcode(ser, grid_gcode)
                    print("✅ Test complete. All pins actuated.")
                    
                elif choice == "2":
                    # Reset coordinates before test
                    reset_coordinates(ser, config)
                    move_to_corner_test(config, ser)
                    print("✅ Corner test complete.")
                    
                elif choice == "3":
                    calibrate_home_position(ser, config)
                    
                elif choice == "4":
                    # Quick servo test
                    quick_servo_test(ser)
                
                elif choice == "5":
                    # We already scanned at startup, but this allows rescanning
                    detect_arduino_ports()
                
                elif choice == "6":
                    # Break the inner loop to update configuration
                    print("\nUpdating configuration...")
                    ser.close()
                    break
                    
                elif choice == "7" or choice.lower() == "q":
                    print("\nExiting program.")
                    ser.close()
                    return  # Exit both loops
                    
                else:
                    print("Invalid choice. Please try again.")
            
        except serial.SerialException as e:
            print(f"❌ Serial Connection Error: {e}")
            print("Try scanning for available ports with option 5")
            
            # Offer port scan option if connection fails
            if input("Scan for available ports now? (y/n): ").lower() == 'y':
                detect_arduino_ports()
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Continue or exit the main loop?
        if input("\nDo you want to run another series of tests? (y/n): ").lower() != 'y':
            print("Exiting program.")
            break

if __name__ == "__main__":
    try:
        full_grid_test()
    except KeyboardInterrupt:
        print("\n\nTest aborted by user.")
        sys.exit(0)
    finally:
        # Clean up servo control
        if 'servo_pwm' in globals():
            servo_pwm.stop()
        GPIO.cleanup()
        print("GPIO pins cleaned up.")
