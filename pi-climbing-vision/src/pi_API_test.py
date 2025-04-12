from typing import List, Dict, Tuple, Optional
import os
import cv2
import numpy as np
import requests
from utils.detection import detect_and_classify_holds
from utils.llm_client import generate_route_description
from utils.camera_helper import setup_camera, capture_image
from config import YOLO_MODEL_PATH, IMAGE_DIR, RESULTS_DIR, LLM_API_URL, USE_PI_CAMERA, RESOLUTION

def main():
    print("=" * 50)
    print("PI CLIMBING VISION - ROUTE ANALYZER")
    print("=" * 50)

    # Ask if user wants to use camera or image from directory
    use_camera = input("Use camera to capture a new image? (y/n, default: n): ").strip().lower() == 'y'
    
    # Ask if user wants custom settings
    use_custom_settings = input("Use custom settings? (y/n, default: n): ").strip().lower() == 'y'
    
    # Set default values
    target_color = 'purple'  # Default color
    sensitivity = 25         # Default sensitivity
    min_area = 200           # Default minimum area
    
    # Process image source (camera or file)
    if use_camera:
        print("\n--- Camera Setup ---")
        use_pi_camera = USE_PI_CAMERA
        resolution = RESOLUTION
        
        if use_custom_settings:
            use_pi_camera = input("Use Pi Camera? (y/n, default: y): ").strip().lower() != 'n'
            resolution_input = input(f"Enter resolution as width,height (default: {RESOLUTION[0]},{RESOLUTION[1]}): ").strip()
            if resolution_input:
                try:
                    width, height = map(int, resolution_input.split(','))
                    resolution = (width, height)
                except:
                    print(f"Invalid format. Using default resolution: {RESOLUTION}")
        
        print(f"Initializing camera (Pi Camera: {use_pi_camera}, Resolution: {resolution})")
        camera = setup_camera(use_picamera=use_pi_camera, resolution=resolution)
        
        if camera is None:
            print("Failed to initialize camera. Exiting.")
            return
        
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        image_path = capture_image(camera, save_path=os.path.join(RESULTS_DIR, "captured_image.jpg"))
        
        if not use_pi_camera and hasattr(camera, 'release'):
            camera.release()
            
        if image_path is None:
            print("Failed to capture image. Exiting.")
            return
            
        print(f"Image captured and saved to: {image_path}")
    else:
        # Fetch an image from the specified folder
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print("No images found in the specified directory.")
            return

        image_path = os.path.join(IMAGE_DIR, image_files[0])  # Get the first image
        print(f"Using image from directory: {image_path}")

    # Get detection parameters
    if use_custom_settings:
        print("\n--- Detection Settings ---")
        # Color selection
        color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]
        print("Available colors:", ", ".join(color_options))
        color_input = input(f"Enter color to detect (default: {target_color}): ").strip().lower()
        if color_input in color_options:
            target_color = color_input
            
        # Sensitivity
        sensitivity_input = input(f"Enter sensitivity level (0-100, default: {sensitivity}): ").strip()
        if sensitivity_input.isdigit():
            sensitivity = max(0, min(100, int(sensitivity_input)))
            
        # Minimum area
        min_area_input = input(f"Enter minimum area to consider as a hold (default: {min_area}): ").strip()
        if min_area_input.isdigit():
            min_area = max(0, int(min_area_input))
    else:
        # Just ask for color if not using custom settings
        color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]
        print("\nAvailable colors:", ", ".join(color_options))
        color_input = input(f"Enter color to detect (default: {target_color}): ").strip().lower()
        if color_input in color_options:
            target_color = color_input

    print(f"\nProcessing image with settings: Color={target_color}, Sensitivity={sensitivity}, Min Area={min_area}")

    # Detect holds and volumes using YOLO
    holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
        image_path,
        target_color=target_color,
        sensitivity=sensitivity,
        min_area=min_area,
        yolo_model_path=YOLO_MODEL_PATH
    )

    if not holds_info:
        print("No holds detected.")
        return

    # Generate a description using the LLM API
    description = generate_route_description(grid_map, use_local_llm=False, api_url=LLM_API_URL)
    print("Generated Route Description:")
    print(description)

    # Optional: Read the description aloud
    read_aloud = input("Read description aloud? (y/n): ").strip().lower() == 'y'
    if read_aloud:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(description)
            engine.runAndWait()
        except ImportError:
            print("Text-to-speech module (pyttsx3) not installed.")
            print("Install with: pip install pyttsx3")
        except Exception as e:
            print(f"Error reading aloud: {e}")

    # Display result image if requested
    show_image = input("Display detection results? (y/n): ").strip().lower() == 'y'
    if show_image:
        try:
            cv2.imshow("Detected Holds", result_image)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying image: {e}")
            print("This might be due to running in a headless environment.")

if __name__ == "__main__":
    main()