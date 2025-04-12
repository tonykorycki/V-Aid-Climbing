from picamera2 import Picamera2
import cv2
import os

def setup_camera(use_picamera=True, resolution=(640, 480)):
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
            image = camera.capture_array()
            cv2.imwrite(save_path, image)
        else:  # OpenCV VideoCapture
            ret, frame = camera.read()
            if ret:
                cv2.imwrite(save_path, frame)
            else:
                return None
        print(f"Image saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None