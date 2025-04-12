'''Running This Code on Raspberry Pi 5
Setup Instructions:

Install required dependencies:

sudo apt update
sudo apt install -y python3-pip python3-opencv libopencv-dev
pip3 install numpy matplotlib torch torchvision ultralytics pillow requests huggingface_hub


For Pi Camera setup:

sudo apt install -y python3-picamera2


For YOLO model:

Transfer your trained model to the Pi or download it
Make sure paths are correctly configured


File structure:

Create a directory for the project
Make sure CV_LLM_integration.py is in a "computer_vision" folder
Put pi-cv-llm.py in the same level as CV_LLM_integration


Potential Challenges:

    Performance limitations:
        YOLO object detection will be slower on the Pi
        The full detection pipeline might take 10-30+ seconds per image
        Local LLM execution could be very slow or may run out of memory

    Memory constraints:
        Even with 8GB RAM, running YOLO + an LLM could cause memory issues
        Consider using an API for LLM functionality instead of local models

    Dependencies:
        Some Python packages might need to be built from source
        Torch specifically can be challenging on ARM architecture

    Display issues:
        If running headless (no monitor), matplotlib will need "Agg" backend
        GUI performance may be sluggish with complex visualizations

    Temperature concerns:
        Intensive processing can cause the Pi to heat up and potentially throttle
        Consider adding a cooling solution

Optimizations:

    The code includes optimizations for limited resources:
        Thread limiting for OpenCV
        Memory management settings
        Option to use an API for LLM instead of local processing
        Headless mode support

    If you need more performance:
        Consider using a lighter YOLO model (like YOLOv5n or YOLOv8n)
        Process at lower resolutions
        Use external processing for the LLM component

The script is designed to alert you to performance issues and provide fallbacks where possible.
'''

import os
import sys
import time
import subprocess
from typing import Dict, Optional
import numpy as np
import cv2

# Import original code functionality
# Add parent directory to path to find CV_LLM_integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import directly since files are at the same level
try:
    from CV_LLM_integration import (
        detect_and_classify_holds,
        load_model,
        predict_route_difficulty,
        visualize_grid_map,
        save_results,
        generate_route_description,
        generate_generic_description,
        analyze_route_complexity
    )
except ImportError:
    print("Error: Could not import CV_LLM_integration module")
    print("Make sure CV_LLM_integration.py is in the same directory as this script")
    sys.exit(1)
##############################################################################
# RASPBERRY PI SPECIFIC SETUP AND OPTIMIZATIONS
##############################################################################

def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed for Raspberry Pi
    """
    required_packages = [
        "opencv-python",
        "numpy",
        "matplotlib",
        "ultralytics",
        "torch",
        "torchvision",
        "pillow",
        "requests",
        "huggingface_hub"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.split('-')[0])  # Handle packages like opencv-python
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        install = input("Would you like to install them now? (y/n): ").strip().lower()
        if install == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("Dependencies installed successfully!")
                return True
            except Exception as e:
                print(f"Error installing dependencies: {e}")
                return False
        else:
            return False
    
    return True

def optimize_for_raspberry_pi():
    """
    Apply Raspberry Pi specific optimizations
    """
    # Get available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.readlines()
        
        for line in mem_info:
            if 'MemTotal' in line:
                total_mem = int(line.split()[1]) / 1024  # Convert to MB
                break
        
        print(f"Total memory available: {total_mem:.1f} MB")
        
        # Set environment variables to optimize PyTorch/CUDA if applicable
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        
        # Limit OpenCV thread usage
        cv2.setNumThreads(2)
        
        return True
    except Exception as e:
        print(f"Warning: Could not optimize for Raspberry Pi: {e}")
        return False

def setup_camera(use_picamera=True, resolution=(640, 480)):
    """
    Set up camera input - either Pi Camera or USB webcam
    """
    if use_picamera:
        try:
            from picamera2 import Picamera2
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
    Capture an image from the camera
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

def setup_display(headless=False):
    """
    Configure display settings based on whether running headless or with display
    """
    if headless:
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        print("Configured for headless operation (no GUI)")
    else:
        # Check if running in X environment
        if 'DISPLAY' not in os.environ:
            print("No display detected, configuring for headless operation")
            import matplotlib
            matplotlib.use('Agg')
        else:
            print("Display detected, using GUI mode")

##############################################################################
# MAIN PROGRAM EXECUTION
##############################################################################

def main():
    print("=" * 50)
    print("CLIMBING ROUTE ANALYZER FOR RASPBERRY PI")
    print("=" * 50)
    
    # Check and setup dependencies
    if not check_dependencies():
        print("Missing dependencies. Please install required packages and try again.")
        return
    
    # Apply Pi-specific optimizations
    optimize_for_raspberry_pi()
    
    # Setup display mode
    headless = input("Running in headless mode (no display)? (y/n, default: n): ").strip().lower() == 'y'
    setup_display(headless)
    
    # Create output directory
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Choose image source
    use_camera = input("Capture new image with camera? (y/n, default: y): ").strip().lower() != 'n'
    image_path = None
    camera = None
    
    if use_camera:
        # Setup camera
        camera_type = input("Use Pi Camera module? (y/n, default: y): ").strip().lower() != 'n'
        resolution_str = input("Enter resolution as width,height (default: 640,480): ").strip()
        
        if resolution_str:
            try:
                width, height = map(int, resolution_str.split(','))
                resolution = (width, height)
            except:
                resolution = (640, 480)
                print("Invalid resolution format. Using default 640x480.")
        else:
            resolution = (640, 480)
        
        # Initialize camera
        camera = setup_camera(use_picamera=camera_type, resolution=resolution)
        if camera is None:
            print("Failed to initialize camera. Exiting.")
            return
        
        # Capture and save image
        image_path = capture_image(camera, save_path=f"{output_dir}/captured_image.jpg")
        if image_path is None:
            print("Failed to capture image. Exiting.")
            return
    else:
        # Use existing image
        default_img = "climbing_wall.jpg"
        image_path = input(f"Enter path to image (default: {default_img}): ").strip()
        if not image_path:
            image_path = default_img
        
        if not os.path.exists(image_path):
            print(f"Image not found at {image_path}. Exiting.")
            return
    
    # Color selection
    color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]
    print("\nAvailable colors:", ", ".join(color_options))
    target_color = input("Enter color to detect (default: purple): ").strip().lower()
    if not target_color or target_color not in color_options:
        target_color = "purple"
    
    # Detection parameters
    sensitivity_str = input("Enter sensitivity level (0-100, default: 25): ").strip()
    sensitivity = int(sensitivity_str) if sensitivity_str.isdigit() else 25
    sensitivity = max(0, min(100, sensitivity))
    
    min_area_str = input("Enter minimum area to consider as a hold (default: 200): ").strip()
    min_area = int(min_area_str) if min_area_str.isdigit() else 200
    
    # YOLO model path
    print("\nEnter YOLO model path (default: train4/weights/best.pt):")
    yolo_model_path = input().strip()
    if not yolo_model_path:
        yolo_model_path = "train4/weights/best.pt"
    
    # Perform detection with progress indicators for slow Pi processing
    print("\nProcessing image with YOLO detection...")
    print("This may take a while on Raspberry Pi...")
    start_time = time.time()
    
    holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
        image_path,
        target_color=target_color,
        sensitivity=sensitivity,
        min_area=min_area,
        yolo_model_path=yolo_model_path
    )
    
    print(f"Detection completed in {time.time() - start_time:.2f} seconds")
    
    # Difficulty prediction
    use_model = input("Use difficulty prediction model? (y/n, default: y): ").strip().lower() != 'n'
    model = None
    predicted_difficulty = None
    
    if use_model:
        model_path = input("Enter classification model path (leave blank for demo): ").strip()
        if not model_path:
            model_path = None
        
        print("Loading difficulty prediction model...")
        model = load_model(model_path)
        
        if model and len(holds_info) > 0:
            predicted_difficulty = predict_route_difficulty(model, grid_map)
            print(f"\nPredicted route difficulty: {predicted_difficulty}")
    
    print(f"\nDetected {len(holds_info)} holds:")
    for i, hold in enumerate(holds_info):
        print(f"Hold {i+1}: {hold['type']} at grid {hold['grid_position']} (Points: {hold['points_value']})")
    
    # LLM for route description
    use_llm = input("Generate route description using LLM? (y/n, default: n): ").strip().lower() == 'y'
    llm_api_url = None
    
    if use_llm:
        print("Warning: Running LLMs locally on Raspberry Pi can be very slow.")
        llm_type = input("Use local LLM (slow) or API? (local/api, default: api): ").strip().lower()
        if llm_type != "local":
            llm_api_url = input("Enter API URL for LLM service: ").strip()
    
    # Save results with optional LLM description
    save_path = os.path.join(output_dir, "climbing_analysis.txt")
    print(f"\nSaving results to {save_path}...")
    
    txt_path, desc_path = save_results(
        image_path, holds_info, grid_map, result_image, masked_image, cropped_region,
        predicted_difficulty, output_dir, use_llm, llm_api_url
    )
    
    # Display results if not in headless mode
    if not headless:
        try:
            import matplotlib.pyplot as plt
            
            fig = visualize_grid_map(grid_map, 
                title=f"Route Grid Map{f' (Difficulty: {predicted_difficulty})' if predicted_difficulty else ''}")
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original with Detections")
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            
            plt.subplot(1, 2, 2)
            plt.title("Hold Grid Map")
            plt.imshow(grid_map, cmap='Blues', interpolation='nearest')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error displaying results: {e}")
            print("Results saved to files for viewing.")
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}/")
    
    # Cleanup
    if camera and not hasattr(camera, 'capture_array'):  # OpenCV camera
        camera.release()

if __name__ == "__main__":
    main()