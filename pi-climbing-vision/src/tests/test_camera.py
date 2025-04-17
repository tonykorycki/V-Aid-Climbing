import os
import sys
import time
import cv2

# Add parent directory to path to access project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.camera_helper import setup_camera, capture_image
from paths import RESULTS_DIR, USE_PI_CAMERA, RESOLUTION

def test_camera():
    """Test both Pi Camera and USB camera capabilities"""
    print("==== Camera Test ====")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Test both camera types
    camera_types = [
        {"name": "Pi Camera", "use_picamera": True},
        {"name": "USB Camera", "use_picamera": False}
    ]
    
    for cam_type in camera_types:
        print(f"\nTesting {cam_type['name']}...")
        try:
            # Initialize camera
            print(f"Initializing with resolution {RESOLUTION}...")
            camera = setup_camera(use_picamera=cam_type["use_picamera"], resolution=RESOLUTION)
            
            if camera is None:
                print(f"❌ {cam_type['name']} initialization failed")
                continue
                
            print(f"✅ {cam_type['name']} initialized successfully")
            
            # Get camera info
            if not cam_type["use_picamera"]:
                print("Camera properties:")
                print(f"  - Width: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))} px")
                print(f"  - Height: {int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))} px")
                print(f"  - FPS: {int(camera.get(cv2.CAP_PROP_FPS))}")
            else:
                print(f"  - Resolution: {RESOLUTION}")
            
            # Capture test image
            print("Capturing test image...")
            save_path = os.path.join(RESULTS_DIR, f"test_{cam_type['name'].lower().replace(' ', '_')}.jpg")
            image_path = capture_image(camera, save_path=save_path)
            
            # Clean up
            if not cam_type["use_picamera"] and hasattr(camera, 'release'):
                camera.release()
            
            if image_path and os.path.exists(image_path):
                print(f"✅ Test image saved to: {image_path}")
            else:
                print(f"❌ Failed to capture image")
                
        except Exception as e:
            print(f"❌ Error testing {cam_type['name']}: {e}")
            print("This usually means this camera type is not available on your device")
    
    print("\nCamera test complete. If at least one camera works, you can proceed.")
    print("Note: For Pi Camera to work, make sure it's enabled in raspi-config")
    print("      and the camera ribbon cable is properly connected.")

if __name__ == "__main__":
    test_camera()