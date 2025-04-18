import os
import sys
import time
import cv2
import subprocess

# Define constants directly in this file instead of importing
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))
RESOLUTION = (640, 480)

def trigger_autofocus():
    """
    Sends an I2C command to the ArduCam to trigger autofocus.
    Works for ArduCam 16MP AF modules (like IMX519).
    """
    try:
        # These values come from ArduCam's autofocus control spec
        # The address 0x0c is the onboard AF driver, and 0x04 0x00 is the trigger command
        subprocess.run(['i2cset', '-y', '11', '0x0c', '0x04', '0x00'], check=True)
        print("✅ Autofocus triggered.")
    except Exception as e:
        print(f"❌ Failed to trigger autofocus: {e}")

def setup_camera(use_picamera=True, resolution=RESOLUTION):
    """
    Set up camera input - either Pi Camera or USB webcam.
    """
    if use_picamera:
        try:
            from picamera2 import Picamera2
            
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": resolution})
            picam.configure(config)
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
            '''
            ADDED AUTOFOCUS FEATURE AND TIMING FOR REFOCUS
            '''
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.waitKey(1000)
            '''
            ADDED AUTOFOCUS FEATURE AND TIMING FOR REFOCUS
            '''
            if not cap.isOpened():
                print("Failed to open USB camera")
                return None
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
            camera.start()
            time.sleep(2)  # Give camera time to adjust
            image = camera.capture_array()
            cv2.imwrite(save_path, image)
            camera.stop()
        else:  # OpenCV VideoCapture
            ret, frame = camera.read()
            if ret:
                cv2.imwrite(save_path, frame)
            else:
                print("Failed to capture frame from USB camera")
                return None
        return save_path
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def test_camera():
    """Test both Pi Camera and USB camera capabilities"""
    print("==== Camera Test ====")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Writing permissions: {os.access(RESULTS_DIR, os.W_OK)}")
    
    # Test Pi Camera (only if you have one)
    print("\nTesting Pi Camera...")
    try:
        camera = setup_camera(use_picamera=True, resolution=RESOLUTION)
        trigger_autofocus()
        
        if camera is None:
            print("❌ Pi Camera initialization failed")
        else:
            print("✅ Pi Camera initialized successfully")
            
            # For Pi Camera using picamera2
            print("Capturing image...")
            save_path = os.path.join(RESULTS_DIR, "test_pi_camera.jpg")
            try:
                camera.start()
                time.sleep(1)  # Let camera warm up

                print("🔍 Triggering autofocus...")
                camera.set_controls({"AfMode": 1})        # Continuous autofocus mode
                time.sleep(0.2)
                camera.set_controls({"AfTrigger": 0})     # Trigger autofocus
                time.sleep(0.2)

                # Wait until focus completes
                focus_complete = False
                for _ in range(15):  # Try for ~3 seconds
                    metadata = camera.capture_metadata()
                    af_state = metadata.get("AfState", -1)
                    if af_state == 2:  # Focused
                        print("✅ Autofocus complete.")
                        focus_complete = True
                        break
                    time.sleep(0.2)

                if not focus_complete:
                    print("⚠️ Autofocus did not report success in time.")

                # Capture image after focus settles
                camera.capture_file(save_path)
                camera.stop()

                if os.path.exists(save_path):
                    print(f"✅ Test image saved to: {save_path}")
                else:
                    print("❌ Failed to capture image - file not created")
            except Exception as e:
                print(f"❌ Error in capture operation: {e}")
    except Exception as e:
        print(f"❌ Error testing Pi Camera: {e}")
        print("This usually means Pi Camera is not available on your device")
    
    print("\nCamera test complete.")
    print("Note: For Pi Camera to work, make sure it's enabled in raspi-config")
    print("      and the camera ribbon cable is properly connected.")

if __name__ == "__main__":
    test_camera()
