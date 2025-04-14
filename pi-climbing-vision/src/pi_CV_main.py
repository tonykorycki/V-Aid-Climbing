from typing import List, Dict, Tuple, Optional
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import requests
from utils.detection import detect_and_classify_holds, analyze_route_complexity
from utils.grid_mapping import visualize_grid_map, create_route_visualization
from utils.llm_client import generate_route_description
from utils.camera_helper import setup_camera, capture_image
from config import YOLO_MODEL_PATH, IMAGE_DIR, RESULTS_DIR, LLM_API_URL, USE_PI_CAMERA, RESOLUTION

def show_visualizations_non_blocking(image_path, holds_info, grid_map, result_image, masked_image, 
                                    cropped_region, predicted_difficulty=None):
    """
    Display visualizations in a non-blocking way.
    
    Args:
        image_path: Path to the original image
        holds_info: List of hold information dictionaries
        grid_map: 12x12 numpy array representing the route
        result_image: Image with drawn bounding boxes
        masked_image: Masked image showing only detected holds
        cropped_region: (x, y, w, h) of the cropped region
        predicted_difficulty: Optional predicted difficulty
    """
    # Create visualization in a new thread
    def show_viz():
        try:
            fig = create_route_visualization(
                image_path=image_path,
                holds_info=holds_info,
                grid_map=grid_map,
                result_image=result_image,
                masked_image=masked_image,
                cropped_region=cropped_region,
                predicted_difficulty=predicted_difficulty
            )
            
            plt.figure(fig.number)
            plt.suptitle("Route Analysis - Press any key in the figure to close", fontsize=12)
            plt.draw()
            plt.pause(0.001)  # Small pause to ensure rendering
            
            # Block main thread until a key is pressed
            key_press = plt.waitforbuttonpress()
            plt.close(fig)
            
        except Exception as e:
            print(f"Error displaying visualizations: {e}")
            print("This might be due to running in a headless environment.")
    
    # Start visualization in a separate thread so it doesn't block execution
    viz_thread = threading.Thread(target=show_viz)
    viz_thread.daemon = True  # Thread will exit when main program exits
    viz_thread.start()
    
    # Don't wait for the thread to complete

def save_visualization(image_path, holds_info, grid_map, result_image, masked_image, 
                      cropped_region, predicted_difficulty=None, output_dir=None):
    """
    Save visualizations to files without displaying them.
    
    Args:
        image_path: Path to the original image
        holds_info: List of hold information dictionaries
        grid_map: 12x12 numpy array representing the route
        result_image: Image with drawn bounding boxes
        masked_image: Masked image showing only detected holds
        cropped_region: (x, y, w, h) of the cropped region
        predicted_difficulty: Optional predicted difficulty
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save detected image
    cv2.imwrite(f"{output_dir}/{base_name}_detected_{timestamp}.jpg", result_image)
    
    # Save masked image
    cv2.imwrite(f"{output_dir}/{base_name}_masked_{timestamp}.jpg", masked_image)
    
    # Save cropped image
    x, y, w, h = cropped_region
    orig_img = cv2.imread(image_path)
    cropped_img = orig_img[y:y+h, x:x+w]
    cv2.imwrite(f"{output_dir}/{base_name}_cropped_{timestamp}.jpg", cropped_img)
    
    # Create and save grid map visualization
    complexity_metrics = analyze_route_complexity(grid_map)
    title = f"Route Grid Map{f' (Difficulty: {predicted_difficulty})' if predicted_difficulty else ''}"
    fig = visualize_grid_map(grid_map, title=title, complexity_metrics=complexity_metrics)
    fig.savefig(f"{output_dir}/{base_name}_grid_{timestamp}.png")
    plt.close(fig)
    
    # Save hold information and grid map to text/CSV files
    txt_path = f"{output_dir}/{base_name}_holds_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Detected {len(holds_info)} holds:\n")
        f.write(f"Cropped region: x={x}, y={y}, w={w}, h={h}\n")
        if predicted_difficulty:
            f.write(f"Predicted difficulty: {predicted_difficulty}\n")
        f.write("-" * 50 + "\n")
        for i, hold in enumerate(holds_info):
            f.write(f"Hold {i+1}: {hold['type']}\n")
            f.write(f"  Position: {hold['position']}\n")
            f.write(f"  Grid position: {hold['grid_position']}\n")
            f.write(f"  Dimensions: {hold['dimensions']}\n")
            f.write(f"  Area: {hold['area']} pixels\n")
            f.write("-" * 50 + "\n")
    
    # Save grid map data
    np.savetxt(f"{output_dir}/{base_name}_grid_{timestamp}.csv", grid_map, delimiter=',', fmt='%d')
    
    return txt_path

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
    sensitivity = None       # Will be auto-calculated based on image brightness
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
        color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink"]
        print("Available colors:", ", ".join(color_options))
        color_input = input(f"Enter color to detect (default: {target_color}): ").strip().lower()
        if color_input in color_options:
            target_color = color_input
            
        # Sensitivity
        sensitivity_input = input("Enter sensitivity level (0-100, leave blank for auto): ").strip()
        if sensitivity_input.isdigit():
            sensitivity = max(0, min(100, int(sensitivity_input)))
        else:
            print("Using auto-sensitivity based on image brightness")
            
        # Minimum area
        min_area_input = input(f"Enter minimum area to consider as a hold (default: {min_area}): ").strip()
        if min_area_input.isdigit():
            min_area = max(0, int(min_area_input))
    else:
        # Just ask for color if not using custom settings
        color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink"]
        print("\nAvailable colors:", ", ".join(color_options))
        color_input = input(f"Enter color to detect (default: {target_color}): ").strip().lower()
        if color_input in color_options:
            target_color = color_input

    # If sensitivity is still None, calculate it based on image brightness
    if sensitivity is None:
        temp_img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV)
        brightness = hsv_img[:, :, 2].mean()
        sensitivity = int(60 - (brightness / 255) * 40)
        sensitivity = max(10, min(60, sensitivity))
        print(f"Auto-sensitivity set to {sensitivity} based on image brightness")

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
    print("\nGenerated Route Description:")
    print("-" * 50)
    print(description)
    print("-" * 50)

    # Save results
    output_path = save_visualization(
        image_path=image_path,
        holds_info=holds_info,
        grid_map=grid_map,
        result_image=result_image,
        masked_image=masked_image,
        cropped_region=cropped_region,
        output_dir=RESULTS_DIR
    )
    print(f"\nResults saved to: {output_path}")

    # Optional: Read the description aloud
    read_aloud = input("Read description aloud? (y/n, default: n): ").strip().lower() == 'y'
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

    # Display visualizations in a non-blocking way if requested
    show_image = input("Display detection results? (y/n, default: y): ").strip().lower() != 'n'
    if show_image:
        show_visualizations_non_blocking(
            image_path=image_path,
            holds_info=holds_info,
            grid_map=grid_map,
            result_image=result_image,
            masked_image=masked_image,
            cropped_region=cropped_region
        )
        print("Visualization window opened. Press any key in the figure to close it.")
        print("The program will continue running.")

    # Ask if user wants to save the route description
    save_description = input("Save route description to a file? (y/n, default: y): ").strip().lower() != 'n'
    if save_description:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        desc_path = f"{RESULTS_DIR}/{base_name}_description_{timestamp}.txt"
        with open(desc_path, 'w') as f:
            f.write(description)
        print(f"Description saved to: {desc_path}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()