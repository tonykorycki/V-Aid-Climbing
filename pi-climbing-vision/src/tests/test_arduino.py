import serial
import time
import os
import sys
import random
import numpy as np

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Grid settings
GRID_WIDTH = 12
GRID_HEIGHT = 12
GRID_SPACING = 10  # mm

# Arduino serial settings
ARDUINO_PORT = '/dev/ttyUSB0'
ARDUINO_BAUD = 115200

def generate_test_grid():
    """ Generate a 12x12 grid with ~3 '1's and 2 '2's """
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    ones = random.sample(range(GRID_HEIGHT * GRID_WIDTH), 3)
    twos = random.sample([i for i in range(GRID_HEIGHT * GRID_WIDTH) if i not in ones], 2)

    for idx in ones:
        y, x = divmod(idx, GRID_WIDTH)
        grid[y, x] = 1
    for idx in twos:
        y, x = divmod(idx, GRID_WIDTH)
        grid[y, x] = 2

    return grid

def generate_relative_gcode(grid):
    """ Generate G-code using relative positioning """
    gcode = ["G21 ; Set units to mm", "G91 ; Relative positioning"]
    current_x, current_y = 0, 0

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            val = grid[y, x]
            if val in [1, 2]:
                target_x = x * GRID_SPACING
                target_y = y * GRID_SPACING
                dx = target_x - current_x
                dy = target_y - current_y
                gcode.append(f"G0 X{dx} Y{dy} F3000")
                gcode.append("M3 S255 ; Activate actuator")
                gcode.append("G4 P0.5 ; Dwell 0.5s")
                gcode.append("M5 ; Deactivate actuator")
                current_x += dx
                current_y += dy

    return "\n".join(gcode)

def generate_absolute_gcode(grid):
    """ Generate G-code using absolute positioning """
    
    # Reduce grid scaling if hitting boundaries
    scaling_factor = 10  # Adjust this if needed (was GRID_SPACING)
    
    gcode = [
        "G21 ; Set units to mm", 
        "G90 ; Absolute positioning",
        "G0 X0 Y0 F1500 ; Home to origin",
        "G4 P1 ; Dwell 1s to ensure we're at origin"
    ]
    
    # Find all cells with values 1 or 2
    points = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if grid[y, x] in [1, 2]:
                points.append((x, y, grid[y, x]))
    
    print(f"Found {len(points)} active points in grid")
    
    # Visit each point
    for idx, (x, y, val) in enumerate(points):
        abs_x = x * scaling_factor
        abs_y = y * scaling_factor
        gcode.append(f"; Moving to point {idx+1}/{len(points)}: ({x},{y}) - value: {val}")
        gcode.append(f"G0 X{abs_x} Y{abs_y} F1500 ; Move to point")
        gcode.append("G4 P1 ; Dwell 1s to ensure arrival")
        gcode.append("M3 S255 ; Activate actuator")
        gcode.append("G4 P2 ; Dwell 2s with actuator active")
        gcode.append("M5 ; Deactivate actuator")
        gcode.append("G4 P1 ; Dwell 1s after deactivating")
    
    # Return to origin with explicit command
    gcode.append("; Returning to origin")
    gcode.append("G0 X0 Y0 F1500 ; Move back to origin")
    gcode.append("G4 P2 ; Dwell 2s at origin")
    
    return "\n".join(gcode)

def send_gcode(ser, gcode):
    """ Send G-code line by line over serial with longer delays """
    lines = gcode.split("\n")
    total_lines = len(lines)
    
    for i, line in enumerate(lines):
        print(f"Sending ({i+1}/{total_lines}): {line}")
        ser.write((line + "\n").encode())
        
        # Skip comments when adding extra delay
        
            # Give more time for movement commands
        if "G0" in line or "G1" in line:
            print("  Movement command - waiting 2 seconds...")
            time.sleep(2.0)  # Much longer delay for movement
        elif "M3" in line or "M5" in line:
            print("  Actuator command - waiting 1 second...")
            time.sleep(1.0)  # Longer delay for actuator
        else:
            print("  Standard command - waiting 0.5 seconds...")
            time.sleep(0.5)  # Standard delay
          
        # Flush buffer and wait for Arduino to process

def test_arduino_connection():
    print("==== Arduino Connection & G-code Test ====")
    common_ports = ['/dev/ttyUSB0', '/dev/ttyACM0']
    detected_port = None

    for port in common_ports:
        try:
            with serial.Serial(port, 115200, timeout=1) as _:
                detected_port = port
                print(f"✓ Arduino detected on {port}")
                break
        except:
            pass

    arduino_port = detected_port if detected_port else input(f"Enter Arduino port (default: {ARDUINO_PORT}): ").strip() or ARDUINO_PORT
    baud_input = input(f"Enter baud rate (default: {ARDUINO_BAUD}): ").strip()
    arduino_baud = int(baud_input) if baud_input.isdigit() else ARDUINO_BAUD

    print(f"Connecting to Arduino on {arduino_port} at {arduino_baud} baud...")

    try:
        ser = serial.Serial(arduino_port, arduino_baud, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        print("✓ Serial connection established!")

        # Generate and send test G-code
        grid = generate_test_grid()
        print("Generated Test Grid:\n", grid)
        # Use the new function here
        gcode = generate_absolute_gcode(grid)
        send_gcode(ser, gcode)

        print("\n✓ G-code test complete!")
        return True

    except serial.SerialException as e:
        print(f"\n✗ Serial port error: {e}")
        print("Try checking cable, port, or permissions.")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_arduino_connection()
