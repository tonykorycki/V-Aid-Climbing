import RPi.GPIO as GPIO
import time

# Button GPIO pin configurations
CYCLE_BUTTON_PIN = 17
SELECT_BUTTON_PIN = 27

def test_buttons():
    # Setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(CYCLE_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SELECT_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    print("Button Test - Press buttons to see status (Press Ctrl+C to exit)")
    print(f"Cycle button: GPIO {CYCLE_BUTTON_PIN}, Select button: GPIO {SELECT_BUTTON_PIN}")
    print("With pull-ups, button press = LOW (0), released = HIGH (1)")
    
    try:
        while True:
            cycle_state = GPIO.input(CYCLE_BUTTON_PIN)
            select_state = GPIO.input(SELECT_BUTTON_PIN)
            
            cycle_status = "PRESSED" if cycle_state == GPIO.LOW else "released"
            select_status = "PRESSED" if select_state == GPIO.LOW else "released"
            
            print(f"Cycle: {cycle_status} | Select: {select_status}    ", end="\r")
            
            # If button pressed, print on new line for visibility
            if cycle_state == GPIO.LOW or select_state == GPIO.LOW:
                print(f"Cycle: {cycle_status} | Select: {select_status}    ")
                time.sleep(0.5)
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nButton test ended.")
    finally:
        GPIO.cleanup()
        print("GPIO pins cleaned up.")

if __name__ == "__main__":
    test_buttons()