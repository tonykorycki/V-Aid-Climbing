import os
import sys
import time
import cv2
import subprocess

# Define constants
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))
RESOLUTION = (640, 480)

def setup_camera(use_picamera=True, resolution=RESOLUTION):
    """
    Set up PiCamera2 or USB camera with max sharpness autofocus logic.
    """
    if use_picamera:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": resolution})
            picam.configure(config)
            return picam
        except ImportError:
            print("⚠️ PiCamera2 not found. Falling back to USB camera.")
            use_picamera = False
        except Exception as e:
            print(f"⚠️ Error initializing PiCamera2: {e}")
            use_picamera = False

    if not use_picamera:
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print("⏳ Warming up USB camera with extra frames...")
            for _ in range(15):
                ret, _ = cap.read()
                time.sleep(0.3)  # Slower = more time to refocus
            return cap
        except Exception as e:
            print(f"❌ USB camera init failed: {e}")
            return None

def capture_image(camera, save_path="captured_image.jpg"):
    """
    Capture a super sharp image from PiCamera2 or USB webcam.
    """
    try:
        if hasattr(camera, 'capture_array'):  # PiCamera2 logic
            camera.start()
            time.sleep(1.5)

            print("🔍 Triggering PiCamera2 autofocus...")
            camera.set_controls({"AfMode": 1})        # Continuous
            time.sleep(0.2)
            camera.set_controls({"AfTrigger": 0})     # Manual trigger
            time.sleep(3.0)                            # Long delay to settle

            # Optional: Debug autofocus state
            metadata = camera.capture_metadata()
            print("📊 Autofocus metadata:", metadata)

            # Lock autofocus
            camera.set_controls({"AfMode": 0})
            time.sleep(0.2)

            camera.capture_file(save_path)
            camera.stop()
            print(f"✅ Image captured: {save_path}")
            return save_path

        else:  # OpenCV (USB) logic
            print("🔍 USB autofocus settle phase...")
            time.sleep(3.0)

            # Discard extra frames before final capture
            for _ in range(7):
                ret, _ = camera.read()
                time.sleep(0.1)

            ret, frame = camera.read()
            if ret:
                cv2.imwrite(save_path, frame)
                print(f"✅ USB image captured: {save_path}")
                return save_path
            else:
                print("❌ USB capture failed")
                return None

    except Exception as e:
        print(f"❌ Capture error: {e}")
        return None

def test_camera():
    """Full autofocus + capture test."""
    print("==== SUPER SHARP CAMERA TEST ====")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    print(f"🗂️ Results dir: {RESULTS_DIR} (Writable: {os.access(RESULTS_DIR, os.W_OK)})")

    camera = setup_camera(use_picamera=True, resolution=RESOLUTION)
    if camera is None:
        print("❌ No camera initialized.")
        return

    save_path = os.path.join(RESULTS_DIR, "max_sharpness_test.jpg")
    final_path = capture_image(camera, save_path)

    if final_path and os.path.exists(final_path):
        print("✅ Sharp image saved successfully.")
    else:
        print("❌ Image capture failed.")

if __name__ == "__main__":
    test_camera()
