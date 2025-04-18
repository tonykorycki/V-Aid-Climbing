import os
import sys
import time
import cv2
import subprocess

# Constants
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))
RESOLUTION = (1920, 1080)

def setup_camera(use_picamera=True, resolution=RESOLUTION):
    """
    Initialize PiCamera2 or fallback USB cam with extended warmup and autofocus.
    """
    if use_picamera:
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": resolution})
            picam.configure(config)
            return picam
        except ImportError:
            print("⚠️ PiCamera2 not found. Using USB cam.")
            use_picamera = False
        except Exception as e:
            print(f"⚠️ PiCamera2 error: {e}")
            use_picamera = False

    if not use_picamera:
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            print("⏳ USB warmup: 30 frames, slower pace for max stability...")
            for _ in range(30):
                ret, _ = cap.read()
                time.sleep(0.25)

            return cap
        except Exception as e:
            print(f"❌ USB camera failed: {e}")
            return None

def capture_image(camera, save_path="captured_image.jpg"):
    """
    Capture a super-sharp image from either PiCamera2 or USB cam.
    """
    try:
        if hasattr(camera, 'capture_array'):  # PiCamera2
            camera.start()
            time.sleep(2)

            print("🔍 Triggering PiCamera2 autofocus...")
            camera.set_controls({"AfMode": 1})
            time.sleep(0.5)
            camera.set_controls({"AfTrigger": 0})
            time.sleep(5.0)  # LONG focus settle time

            # Optional debug
            metadata = camera.capture_metadata()
            print("📊 Autofocus metadata:", metadata)

            # Lock focus
            camera.set_controls({"AfMode": 0})
            time.sleep(0.5)

            print("📸 Capturing ultra-sharp image...")
            camera.capture_file(save_path)
            time.sleep(0.5)
            camera.stop()

            print(f"✅ Image saved: {save_path}")
            return save_path

        else:  # USB cam
            print("🔍 USB settle delay: 5s")
            time.sleep(5)

            print("📸 Discarding 15 pre-capture frames...")
            for _ in range(15):
                ret, _ = camera.read()
                time.sleep(0.15)

            print("📸 Taking final sharp frame...")
            ret, frame = camera.read()
            if ret:
                cv2.imwrite(save_path, frame)
                print(f"✅ USB image saved: {save_path}")
                return save_path
            else:
                print("❌ Failed to read frame.")
                return None

    except Exception as e:
        print(f"❌ Capture error: {e}")
        return None

def test_camera():
    print("===== ULTRA SHARP CAMERA TEST =====")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    print(f"🗂️ Saving to: {RESULTS_DIR} (Writable: {os.access(RESULTS_DIR, os.W_OK)})")

    camera = setup_camera(use_picamera=True, resolution=RESOLUTION)
    if camera is None:
        print("❌ Camera not initialized.")
        return

    save_path = os.path.join(RESULTS_DIR, "ultra_sharp_test.jpg")
    final_path = capture_image(camera, save_path)

    if final_path and os.path.exists(final_path):
        print("✅ Ultra-sharp capture complete!")
    else:
        print("❌ Image capture failed.")

if __name__ == "__main__":
    test_camera()
