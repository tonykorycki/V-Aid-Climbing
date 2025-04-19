import serial
import time

# ===== Global Configurable Constants =====
GRID_WIDTH = 12
GRID_HEIGHT = 12
GRID_SPACING = 10  # mm between each pin
OFFSET_X = 5       # mm from X=0 to first column
OFFSET_Y = 10      # mm from Y=0 to first row
FEEDRATE = 3000    # Movement speed
SERVO_PUSH_DELAY = 0.5  # seconds to hold actuator
SERVO_RETRACT_DELAY = 0.5  # seconds after actuator retract
ARDUINO_BAUD = 115200

ARDUINO_PORT = '/dev/ttyUSB0'
# ARDUINO_PORT = 'COM4'  # ← for example, on Windows


# ===== G-code Logic =====
def generate_full_grid_gcode():
    """Generate G-code to sequentially push all 144 pins on a 12x12 grid."""
    gcode = [
        "G21 ; Use millimeters",
        "G90 ; Absolute positioning"
    ]

    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            x = OFFSET_X + col * GRID_SPACING
            y = OFFSET_Y + row * GRID_SPACING
            gcode.append(f"G0 X{x} Y{y} F{FEEDRATE} ; Move to ({col},{row})")
            gcode.append("M3 S255 ; Push pin (servo activate max)")
            gcode.append(f"G4 P{SERVO_PUSH_DELAY} ; Wait while pushing")
            gcode.append("M5 ; Retract actuator")
            gcode.append(f"G4 P{SERVO_RETRACT_DELAY} ; Wait after retract")

    gcode.append("G0 X0 Y0 F3000 ; Return to origin")
    return "\n".join(gcode)

# ===== Serial Communication =====
def send_gcode(ser, gcode):
    print("→ Flushing and waking Arduino...")
    ser.write(b"\r\n\r\n")
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
                print(f"  ↳ Arduino: {response}")

# ===== Main Script =====
def full_grid_test():
    print("==== Full 12x12 Grid Actuation Test ====")
    gcode = generate_full_grid_gcode()
    try:
        print(f"Connecting to Arduino on {ARDUINO_PORT}...")
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)
        print("✓ Serial connection established.")
        send_gcode(ser, gcode)
        print("✅ Test complete. All pins actuated.")
        ser.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    full_grid_test()
