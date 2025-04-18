import os
import sys
import time
import cv2
import subprocess

# Define constants
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))
RESOLUTION = (640, 480)

def trigger_autofocus_i2c():
    """
    Sends an I2C command to the ArduCam to trigger autofocus.
    """
    try:
        subprocess.run(['i2cset', '-y', '11', '0x0c', '0x04', '0x00'], check=True)
        print("✅ I2C Autofocus triggered.")
    except Exception as e:
        print(f"❌ Failed to trigger I2C autofocus: {e}")

def setup_camera(use_picamera=True, resolution=RESOLUTION):
    """
    Set up either PiCamera2 or USB camera with autofocus logic.
    """
    if use_picamera:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": resolution})
            picam.configure(config)
            return picam
        except ImportError:
            print("PiCamera2 not found. Falling back to USB camera.")
            use_picamera = False
        except Exception as e:
            print(f"Error initializing PiCamera2: {e}")
            use_picamera = False

    if not use_picamera:
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print("⏳ Warming up USB camera and triggering autofocus...")
            for _ in range(10):  # Read 10 frames to let autofocus adjust
                ret, _ = cap.read()
                time.sleep(0.3)
            return cap
        except Exception as e:
            print(f"USB camera failed: {e}")
            return None

def capture_image(camera, save_path="captured_image.jpg"):
    """
    Capture and save an image after triggering autofocus.
    """
    try:
        if hasattr(camera, 'capture_array'):  # PiCamera2 logic
            camera.start()
            time.sleep(1)
            print("🔍 Triggering PiCamera2 autofocus...")

            camera.set_controls({"AfMode": 1})        # Continuous
            time.sleep(0.2)
            camera.set_controls({"AfTrigger": 0})     # Trigger
            time.sleep(2.0)                            # Let it focus

            # Optional: lock focus
            camera.set_controls({"AfMode": 0})

            # Debug autofocus state
            metadata = camera.capture_metadata()
            print("AF Metadata:", metadata)

            # Capture and save
            camera.capture_file(save_path)
            camera.stop()
            print(f"✅ Image saved: {save_path}")
            return save_path

        else:  # OpenCV camera logic
            print("📸 Capturing from USB webcam...")
            time.sleep(2)  # Let autofocus settle
            for _ in range(5):  # Extra frames for sharpness
                ret, frame = camera.read()
                time.sleep(0.1)

            ret, frame = camera.read()
            if ret:
                cv2.imwrite(save_path, frame)
                print(f"✅ Image saved: {save_path}")
                return save_path
            else:
                print("❌ Failed to capture from webcam.")
                return None

    except Exception as e:
        print(f"❌ Capture error: {e}")
        return None

def test_camera():
    """Test autofocus and image capture."""
    print("==== Testing Camera Autofocus ====")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    print(f"Results dir: {RESULTS_DIR} (Writable: {os.access(RESULTS_DIR, os.W_OK)})")

    camera = setup_camera(use_picamera=True, resolution=RESOLUTION)
    if camera is None:
        print("❌ No camera initialized.")
        return

    save_path = os.path.join(RESULTS_DIR, "autofocus_test.jpg")
    captured_path = capture_image(camera, save_path)
    if captured_path and os.path.exists(captured_path):
        print("✅ Autofocus test completed successfully.")
    else:
        print("❌ Autofocus test failed to produce image.")

if __name__ == "__main__":
    test_camera()
