import serial
import time
import os
import sys

# Add parent directory to path to access paths.py if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Arduino serial connection
ARDUINO_PORT = '/dev/ttyUSB0'  # Default for most USB Arduino connections
ARDUINO_BAUD = 115200

def test_arduino_connection():
    print("==== Arduino Connection Test ====")
    
    # Common Arduino ports on Raspberry Pi
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
    
    baud = input(f"Enter baud rate (default: {ARDUINO_BAUD}): ").strip()
    if baud.isdigit():
        arduino_baud = int(baud)
    else:
        arduino_baud = ARDUINO_BAUD
    
    print(f"Attempting to connect to Arduino on {arduino_port} at {arduino_baud} baud")
    
    try:
        # Try opening the connection
        with serial.Serial(arduino_port, arduino_baud, timeout=5) as ser:
            print("✓ Serial connection established!")
            
            # Send test commands
            test_commands = [
                "M115",    # Get firmware info
                "G28",     # Home all axes
                "G0 X10 Y10 F1000",  # Move to position
                "G0 X0 Y0 F1000"     # Return to origin
            ]
            
            for cmd in test_commands:
                print(f"Sending: {cmd}")
                ser.write(f"{cmd}\n".encode())
                time.sleep(1)
                
                # Read response (may vary based on firmware)
                response = ""
                start_time = time.time()
                while time.time() - start_time < 3:  # 3 second timeout
                    if ser.in_waiting:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        response += line + "\n"
                        if "ok" in line.lower():
                            break
                    time.sleep(0.1)
                
                print(f"Response: {response if response else 'No response (may be normal)'}")
                
            print("\n✓ Arduino communication test complete!")
            print("Note: For 3D printer firmware, responses should include 'ok'")
            print("For custom Arduino firmware, responses may vary")
            return True
            
    except serial.SerialException as e:
        print(f"\n✗ Error opening serial port: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the Arduino is connected to the USB port")
        print("2. Check available ports:")
        
        # Show available ports based on platform
        if sys.platform.startswith('linux'):
            print("   Run: ls -l /dev/tty*")
            os.system("ls -l /dev/tty*")
        elif sys.platform.startswith('win'):
            print("   Check Device Manager > Ports (COM & LPT)")
        elif sys.platform.startswith('darwin'):
            print("   Run: ls -l /dev/cu.*")
            os.system("ls -l /dev/cu.*")
            
        print("3. Verify the Arduino has GCODE-compatible firmware installed")
        print("4. Try a different USB cable")
        print("5. Ensure you have permission to access the port (may need to run as sudo)")
        return False
        
    except Exception as e:
        print(f"\n✗ Error during communication: {e}")
        return False

if __name__ == "__main__":
    test_arduino_connection()