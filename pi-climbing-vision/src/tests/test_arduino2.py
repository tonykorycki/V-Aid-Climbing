import serial
import time
import os
import sys
import random
import numpy as np

# Grid settings
GRID_WIDTH = 12
GRID_HEIGHT = 12
GRID_SPACING = 10  # mm

# Arduino serial settings
ARDUINO_PORT = '/dev/ttyUSB0'
ARDUINO_BAUD = 115200

def generate_test_grid():
    """Generate a 12x12 grid with exactly 3 '1's and 2 '2's in different spots."""
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    total_cells = GRID_WIDTH * GRID_HEIGHT
    spots = random.sample(range(total_cells), 5)
    ones, twos = spots[:3], spots[3:]

    for idx in ones:
        y, x = divmod(idx, GRID_WIDTH)
        grid[y, x] = 1
    for idx in twos:
        y, x = divmod(idx, GRID_WIDTH)
        grid[y, x] = 2

    return grid

def generate_absolute_gcode(grid):
    """Generate absolute G-code with 5 actuator pushes, Z-lift, and 3s delay between each."""
    gcode = [
        "G21 ; Use millimeters",
        "G90 ; Absolute positioning"
    ]

    move_count = 0

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            val = grid[y, x]
            if val in [1, 2]:
                pos_x = x * GRID_SPACING
                pos_y = y * GRID_SPACING

                gcode.append(f"G0 X{pos_x} Y{pos_y} F3000 ; Move to ({x},{y})")
                gcode.append("G0 Z5 F1000 ; Lift")
                gcode.append("M3 S255 ; Activate actuator")
                gcode.append("G4 P0.5 ; Hold actuator")
                gcode.append("M5 ; Deactivate actuator")
                gcode.append("G0 Z0 F1000 ; Lower")
                gcode.append("G4 P3 ; Wait 3 seconds before next")
                move_count += 1

    gcode.append("G0 X0 Y0 F3000 ; Return to origin")
    assert move_count == 5, f"Expected 5 actuator pushes, got {move_count}"
    return "\n".join(gcode)

def send_gcode(ser, gcode):
    """Send G-code to Arduino line-by-line, waiting for 'ok' after each command."""
    print("→ Flushing and waking Arduino...")
    ser.write(b"\r\n\r\n")  # Wake/reset GRBL or other firmware
    time.sleep(2)
    ser.flushInput()

    for line in gcode.split("\n"):
        line = line.strip()
        if not line:
            continue

        print(f"Sending: {line}")
        ser.write((line + "\n").encode())

        while True:
            response = ser.readline().decode().strip().lower()
            if response == "ok":
                break
            elif response:
                print(f"  ↳ Arduino response: {response}")

def test_arduino_connection():
    print("==== Arduino G-code Test: 5 Unique Moves + Pushes w/ Z Lift and Delay ====")

    grid = generate_test_grid()
    print("Generated Grid:\n", grid)
    coords = np.argwhere(grid > 0)
    print("🧠 5 Actuator Coordinates (y,x):", coords.tolist())
    assert len(coords) == 5, "❌ Grid does not contain exactly 5 hold points!"

    gcode = generate_absolute_gcode(grid)

    try:
        print(f"Connecting to Arduino on {ARDUINO_PORT} at {ARDUINO_BAUD} baud...")
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)
        print(f"✓ Serial connection established.")
        send_gcode(ser, gcode)
        print("\n✅ G-code complete. All 5 moves, pushes, and waits sent successfully.")
    except Exception as e:
        print(f"\n❌ Connection failed or error during G-code send: {e}")

if __name__ == "__main__":
    test_arduino_connection()
