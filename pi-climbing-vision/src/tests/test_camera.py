import os
import sys
import time
import cv2

# Define constants directly in this file instead of importing
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))
RESOLUTION = (640, 480)

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
            prev_brightness = None
            stable_count = 0
            max_checks = 30  # max frame tries
            tolerance = 5    # brightness must stay within ±5 for stability

            for _ in range(max_checks):
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()

                if prev_brightness is not None and abs(brightness - prev_brightness) < tolerance:
                    stable_count += 1
                else:
                    stable_count = 0  # reset if not stable

                prev_brightness = brightness
                if stable_count >= 5:  # require 5 consistent frames
                    print("✅ Autofocus & brightness stabilized.")
                    break

                cv2.waitKey(100)  # wait between checks
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
time.sleep(1)

# Trigger autofocus
print("🔍 Triggering autofocus...")
camera.set_controls({"AfMode": 1})  # 1 = Auto
time.sleep(0.5)
camera.set_controls({"AfTrigger": 0})  # 0 = Start trigger
time.sleep(2)  # Allow focus to settle

# Optional: wait until autofocus completes (status is 2 = focused)
focus_done = False
for _ in range(10):
    metadata = camera.capture_metadata()
    af_state = metadata.get("AfState")
    if af_state == 2:
        print("✅ Autofocus complete.")
        focus_done = True
        break
    time.sleep(0.2)

if not focus_done:
    print("⚠️ Autofocus did not report completion in time.")

camera.capture_file(save_path)
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
        
        if camera is None:
            print("❌ Pi Camera initialization failed")
        else:
            print("✅ Pi Camera initialized successfully")
            
            # For Pi Camera using picamera2
            print("Capturing image...")
            save_path = os.path.join(RESULTS_DIR, "test_pi_camera.jpg")
            try:
                camera.start()
                time.sleep(2)  # Give camera time to adjust
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
