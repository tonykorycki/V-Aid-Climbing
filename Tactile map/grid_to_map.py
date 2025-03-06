import numpy as np
import serial
import time

# Define grid size (10x24 tactile map)
GRID_WIDTH = 24
GRID_HEIGHT = 10
GRID_SPACING = 10  # mm between tactile dots

# Serial connection to Arduino (adjust port)
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # Wait for connection

def map_to_grid(holds, width=GRID_WIDTH, height=GRID_HEIGHT):
    """ Convert climbing holds to closest grid points """
    grid = np.zeros((height, width), dtype=int)
    
    for hold in holds:
        x, y = hold["position"]
        grid_x = min(round(x / GRID_SPACING), width - 1)
        grid_y = min(round(y / GRID_SPACING), height - 1)
        grid[grid_y, grid_x] = 1  # Mark hold position
    
    return grid

def generate_gcode(grid):
    """ Generate G-code for XY movement and actuation """
    gcode_commands = ["G21 ; Set units to mm", "G90 ; Absolute positioning"]
    
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if grid[y, x] == 1:
                gcode_commands.append(f"G0 X{x * GRID_SPACING} Y{y * GRID_SPACING} F3000")
                gcode_commands.append("M3 S255 ; Activate actuator (raise dot)")
                gcode_commands.append("G4 P0.5 ; Wait 0.5s")
                gcode_commands.append("M5 ; Deactivate actuator")

    gcode_commands.append("G0 X0 Y0 ; Return to home position")
    return "\n".join(gcode_commands)

def send_gcode(gcode):
    """ Send G-code to motor controller via serial """
    for line in gcode.split("\n"):
        print(f"Sending: {line}")
        ser.write((line + "\n").encode())
        time.sleep(0.1)

# Example climbing holds from CV
climbing_holds = [
    {"type": "foothold", "position": (10, 10)},
    {"type": "crimp", "position": (50, 30)},
    {"type": "sloper", "position": (100, 50)}
]

# Convert hold positions to grid
grid_data = map_to_grid(climbing_holds)

# Generate G-code
gcode_script = generate_gcode(grid_data)

# Send to CNC system
send_gcode(gcode_script)
