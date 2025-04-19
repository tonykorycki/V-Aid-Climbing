from picamera2 import Picamera2
import cv2
import os
import time

RESOLUTION = (2560, 1440)
def setup_camera(use_picamera=True, resolution=RESOLUTION):
    """
    Set up camera input - either Pi Camera or USB webcam.
    """
    if use_picamera:
        try:
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": resolution})
            picam.configure(config)
            picam.start()
            return picam
        except ImportError:
            print("PiCamera2 module not found, falling back to USB camera")
            use_picamera = False
        except Exception as e:
            print(f"Error initializing Pi Camera: {e}")
            print("Falling back to USB camera")
            use_picamera = False
    
    if not use_picamera:
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            print("⏳ USB warmup: 15 frames, slower pace for max stability...")
            for _ in range(15):
                ret, _ = cap.read()
                time.sleep(0.25)

            return cap
        except Exception as e:
            print(f"Error initializing USB camera: {e}")
            return None

def capture_image(camera, save_path="captured_image.jpg"):
    """
    Capture an image from the camera.
    """
    try:
        if hasattr(camera, 'capture_array'):  # PiCamera2
            camera.start()
            time.sleep(2)

            
            camera.set_controls({"AfMode": 1})
            time.sleep(0.5)
            camera.set_controls({"AfTrigger": 0})
            time.sleep(5.0)  # LONG focus settle time

            # Lock focus
            camera.set_controls({"AfMode": 0})
            time.sleep(0.5)

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
        print(f"Error capturing image: {e}")
        return None