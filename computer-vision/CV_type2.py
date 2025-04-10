import os
import sys
import requests
import json
from typing import List, Dict, Tuple, Optional
import numpy as np

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

##############################################################################
# YOLO + COLOR-BASED HOLD/Volume DETECTION WITH GRID MAPPING
##############################################################################

#New Wall Area Function
def detect_wall_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        height, width = image.shape[:2]
        return np.ones((height, width), dtype=np.uint8), (0, 0, width, height)

    largest_contour = max(contours, key=cv2.contourArea)
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(largest_contour)

    if contour_area < 0.1 * image_area:
        height, width = image.shape[:2]
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            line_image = np.zeros((height, width), dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            line_image = cv2.dilate(line_image, kernel, iterations=2)
            line_contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if line_contours:
                largest_contour = max(line_contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)

    return mask, (x, y, w, h)

def detect_and_classify_holds(image_path,
                              target_color='red',
                              sensitivity=25,
                              min_area=200,
                              yolo_model_path="computer-vision/train4/weights/best.pt"):
    """
    Uses a YOLO model to detect objects. For each detected bounding box:
      - If the predicted class is "volume" (or if it's a hold but very large), then we mark it as volume.
      - For non-volume holds, we perform a color check (using your HSV ranges) to decide if it's valid.
      - Valid detections are then mapped onto a 12×12 grid.
    
    Returns:
      holds_info: list of dicts with detected object info.
      grid_map: 12×12 numpy array with grid points (1 or 2) indicating the hold.
      masked_image: image showing only regions that passed the color check (or are volumes).
      result_image: original image annotated with bounding boxes and labels.
      cropped_region: (crop_x, crop_y, crop_w, crop_h) covering all valid detections.
    """
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    orig_h, orig_w = image.shape[:2]

    # Step 1: Detect wall area to isolate from background
    wall_mask, wall_region = detect_wall_area(image)
    x_wall, y_wall, w_wall, h_wall = wall_region

    # Crop image to wall area for further processing
    wall_image = image[y_wall:y_wall+h_wall, x_wall:x_wall+w_wall]

    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return [], np.zeros((12, 12), dtype=np.int32), image, image, (0, 0, orig_w, orig_h)

    # Run YOLO on the image (adjust confidence threshold if needed)
    results = yolo_model.predict(source=wall_image, conf=0.25)
    if len(results) == 0 or len(results[0].boxes) == 0:
        print("No holds detected by YOLO.")
        return [], np.zeros((12, 12), dtype=np.int32), image, image, (0, 0, orig_w, orig_h)

    # Define HSV color ranges (same as your demo code)
    v_base = 50 - sensitivity // 4  # more forgiving with brightness
    s_base = 100 - sensitivity // 4

    color_ranges = {
        'red':   (np.array([0, 150, 80]),   np.array([6, 255, 255])),      # Narrow red to avoid orange
        'red2':  (np.array([174, 150, 80]), np.array([180, 255, 255])),
        'blue': (np.array([90, s_base, v_base]), np.array([130, 255, 255])),    # I changed yo blue
        'green': (np.array([40, 50, 40]),   np.array([80, 255, 255])),
        'yellow':(np.array([25, 150, 150]), np.array([35, 255, 255])),     # Tighten yellow slightly
        'orange':(np.array([10, 150, 150]), np.array([16, 255, 255])),     # Narrow orange to avoid yellow overlap
        'purple':(np.array([120, 40, 40]),  np.array([150, 255, 255])),     # As before – blue now is separate
        'black': (np.array([0, 0, 0]),      np.array([180, 60, 40])),
        'white': (np.array([0, 0, 200]),    np.array([180, 50, 255])),       # Lower the V threshold slightly to catch brighter whites
        'pink': (np.array([160, 80, 80]), np.array([175, 255, 255]))
    }
    
    def is_target_color(roi_bgr):
        """Return True if >20% of ROI is within the target color range."""
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        if target_color == 'red':
            lower1, upper1 = color_ranges['red']
            lower2, upper2 = color_ranges['red2']
            mask1 = cv2.inRange(roi_hsv, lower1, upper1)
            mask2 = cv2.inRange(roi_hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif target_color in color_ranges:
            lower, upper = color_ranges[target_color]
            mask = cv2.inRange(roi_hsv, lower, upper)
        else:
            return False
        return (mask > 0).mean() > 0.2

    result_image = image.copy()
    masked_image = np.zeros_like(image)
    holds_info = []
    valid_x, valid_y, valid_x2, valid_y2 = [], [], [], []

    # Loop over YOLO bounding boxes
    for box in results[0].boxes:
        # Extract bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 += x_wall
        y1 += y_wall
        x2 += x_wall
        y2 += y_wall
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < min_area:
            continue

        # Clip coordinates to image boundaries
        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w - 1, x2))
        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # Get predicted class from YOLO (assume 0 = hold, 1 = volume)
        pred_class = int(box.cls[0].item()) if hasattr(box, 'cls') else 0

        # Calculate bounding box area
        box_area = (x2 - x1) * (y2 - y1)

        # Dismiss a volume if its area is more than 25% of the total image area
        if pred_class == 1 and box_area > 0.15 * (orig_w * orig_h):
            continue

        # If object is classified as volume, we always include it.
        # Otherwise, perform the color check.
        if pred_class == 1:
            color_pass = True
        else:
            roi = image[y1:y2, x1:x2]
            color_pass = is_target_color(roi)
        if not color_pass:
            continue

        # Decide hold type and points:
        # If the model says volume, or if it's a hold but very large, treat it as a volume.
        if pred_class == 1 or (pred_class == 0 and box_area > 3000):
            hold_type = "volume"
            points_to_fill = 2
        else:
            hold_type = "hold"
            points_to_fill = 1

        # Record bounding region info (for cropping later)
        valid_x.append(x1); valid_y.append(y1); valid_x2.append(x2); valid_y2.append(y2)
        masked_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]

        # Draw bounding box and label on the result image
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"{hold_type} ({points_to_fill})",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(result_image, (cx, cy), 3, (0, 255, 255), -1)

        holds_info.append({
            "type": hold_type,
            "position": (cx, cy),
            "bbox": (x1, y1, x2, y2),  # store bbox for grid mapping
            "relative_position": None,  # to be filled after cropping
            "grid_position": None,
            "dimensions": (x2 - x1, y2 - y1),
            "area": box_area,
            "points_value": points_to_fill
        })

    if not holds_info:
        print("No valid holds matched the criteria.")
        cropped_region = (0, 0, orig_w, orig_h)
        return holds_info, np.zeros((12, 12), dtype=np.int32), masked_image, result_image, cropped_region

    # Determine cropped region around all valid detections
    min_x = max(0, min(valid_x))
    min_y = max(0, min(valid_y))
    max_x = min(orig_w - 1, max(valid_x2))
    max_y = min(orig_h - 1, max(valid_y2))
    w = max_x - min_x; h = max_y - min_y
    padding_x = int(w * 0.1)
    padding_y = int(h * 0.1)
    crop_x = max(0, min_x - padding_x)
    crop_y = max(0, min_y - padding_y)
    crop_w = min(orig_w - crop_x, w + 2 * padding_x)
    crop_h = min(orig_h - crop_y, h + 2 * padding_y)
    cropped_region = (crop_x, crop_y, crop_w, crop_h)
    cv2.rectangle(result_image, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 2)

    # Build 12×12 grid map
    grid_map = np.zeros((12, 12), dtype=np.int32)
    cell_width = crop_w / 12.0
    cell_height = crop_h / 12.0

    # For each hold, map its center to grid coordinates
    for hold in holds_info:
        cx, cy = hold["position"]
        rel_x = cx - crop_x
        rel_y = cy - crop_y
        if not (0 <= rel_x < crop_w and 0 <= rel_y < crop_h):
            continue
        grid_x = int(rel_x / cell_width)
        grid_y = int(rel_y / cell_height)
        grid_x = max(0, min(11, grid_x))
        grid_y = max(0, min(11, grid_y))
        # Place points based on the hold's shape using the bounding box
        bbox = hold["bbox"]
        fill_grid_points_by_shape(grid_map, grid_x, grid_y, hold["points_value"],
                                  bbox, (crop_x, crop_y, crop_w, crop_h),
                                  cell_width, cell_height)
        hold["relative_position"] = (rel_x, rel_y)
        hold["grid_position"] = (grid_x, grid_y)
    
    # Filter out volumes that are too isolated
    volume_threshold = 0.05 * min(orig_w, orig_h)  # e.g., 10% of the smaller image dimension
    holds_info = filter_isolated_volumes(holds_info, volume_threshold)

    # Make sure to return the results
    return holds_info, grid_map, masked_image, result_image, cropped_region

def fill_grid_points_by_shape(grid_map, grid_x, grid_y, points_to_fill, bbox, crop_region, cell_width, cell_height):
    """
    Fill the grid cell at (grid_x, grid_y) and, if two points are needed, add a second point
    in a direction that reflects the bounding box orientation (horizontal vs. vertical).
    """
    # Place center point
    grid_map[grid_y, grid_x] = points_to_fill
    if points_to_fill == 2:
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        # Decide offset direction based on orientation
        if box_width >= box_height:
            # Horizontal: try to offset to the left; if not available, go right.
            second_x = grid_x - 1 if grid_x > 0 else grid_x + 1
            second_y = grid_y
        else:
            # Vertical: try to offset upward; if not available, go downward.
            second_y = grid_y - 1 if grid_y > 0 else grid_y + 1
            second_x = grid_x
        grid_map[second_y, second_x] = points_to_fill

def filter_isolated_volumes(holds_info, threshold):
    """
    Remove volumes that are isolated (i.e., no other hold is within a given threshold).
    'threshold' can be set as a fraction (or pixels) of the image size.
    """
    filtered = []
    for i, hold in enumerate(holds_info):
        if hold["type"] == "volume":
            cx, cy = hold["position"]
            has_neighbor = False
            for j, other in enumerate(holds_info):
                if i == j:
                    continue
                ox, oy = other["position"]
                dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                if dist < threshold:
                    has_neighbor = True
                    break
            if has_neighbor:
                filtered.append(hold)
        else:
            filtered.append(hold)
    return filtered

##############################################################################
# REMAINING FUNCTIONS: load_model, predict_route_difficulty, visualize, save, main
##############################################################################

def load_model(model_path=None):
    """
    Load a PyTorch classification model that predicts route difficulty from a 12×12 grid.
    If no model_path is provided, a dummy model is created.
    """
    try:
        import torch
        if model_path is None:
            print("No model path provided. Using demo classification model...")
            class ClimbingRouteModel(torch.nn.Module):
                def __init__(self):
                    super(ClimbingRouteModel, self).__init__()
                    self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool2d(2, 2)
                    self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
                    self.fc1 = torch.nn.Linear(32 * 3 * 3, 128)
                    self.fc2 = torch.nn.Linear(128, 64)
                    self.fc3 = torch.nn.Linear(64, 6)  # 6 difficulty classes
                def forward(self, x):
                    x = self.pool(torch.relu(self.conv1(x)))
                    x = self.pool(torch.relu(self.conv2(x)))
                    x = x.view(-1, 32 * 3 * 3)
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            model = ClimbingRouteModel()
            print("Demo classification model created.")
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            print(f"Classification model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None

def predict_route_difficulty(model, grid_map):
    """
    Predict route difficulty based on the 12×12 grid map.
    """
    try:
        import torch
        input_tensor = torch.tensor(grid_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        difficulty_grades = ["V0-V1", "V2-V3", "V4-V5", "V6-V7", "V8-V9", "V10+"]
        return difficulty_grades[predicted.item()]
    except Exception as e:
        print(f"Error predicting route difficulty: {e}")
        return "Unknown"

def visualize_grid_map(grid_map, title="Route Grid Map"):
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.Blues
    plt.imshow(grid_map, cmap=cmap, interpolation='nearest')
    # Draw grid lines
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    for i in range(grid_map.shape[0] + 1):
        plt.axhline(y=i - 0.5, color='gray', linestyle='-', linewidth=0.5)
    for i in range(grid_map.shape[1] + 1):
        plt.axvline(x=i - 0.5, color='gray', linestyle='-', linewidth=0.5)
    # Write cell values
    for i in range(grid_map.shape[0]):
        for j in range(grid_map.shape[1]):
            if grid_map[i, j] > 0:
                plt.text(j, i, str(grid_map[i, j]),
                         ha="center", va="center",
                         color="white" if grid_map[i, j] > 1 else "black")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def save_results(image_path, holds_info, grid_map, result_image, masked_image, cropped_region,
                 predicted_difficulty=None, output_dir="computer-vison/results", 
                 use_llm=True, llm_api_url=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save images
    cv2.imwrite(f"{output_dir}/{base_name}_detected_{timestamp}.jpg", result_image)
    cv2.imwrite(f"{output_dir}/{base_name}_masked_{timestamp}.jpg", masked_image)
    x, y, w, h = cropped_region
    orig_img = cv2.imread(image_path)
    cropped_img = orig_img[y:y+h, x:x+w]
    cv2.imwrite(f"{output_dir}/{base_name}_cropped_{timestamp}.jpg", cropped_img)
    
    # Generate route description using LLM
    route_description = "No route description generated."
    if use_llm:
        print("\nGenerating route description using LLM...")
        route_description = generate_route_description(
            grid_map, 
            difficulty=predicted_difficulty,
            use_local_llm=(llm_api_url is None),
            api_url=llm_api_url
        )
        print("\nRoute Description:")
        print("-" * 50)
        print(route_description)
        print("-" * 50)
    
    # Save hold information
    txt_path = f"{output_dir}/{base_name}_holds_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Detected {len(holds_info)} holds:\n")
        f.write(f"Cropped region: x={x}, y={y}, w={w}, h={h}\n")
        if predicted_difficulty:
            f.write(f"Predicted difficulty: {predicted_difficulty}\n")
        f.write("\nRoute Description:\n")
        f.write(route_description)
        f.write("\n\n")
        f.write("-" * 50 + "\n")
        for i, hold in enumerate(holds_info):
            f.write(f"Hold {i+1}: {hold['type']}\n")
            f.write(f"  Position: {hold['position']}\n")
            f.write(f"  Relative position in crop: {hold['relative_position']}\n")
            f.write(f"  Grid position: {hold['grid_position']}\n")
            f.write(f"  Dimensions: {hold['dimensions']}\n")
            f.write(f"  Area: {hold['area']} pixels\n")
            f.write(f"  Points: {hold.get('points_value', 1)}\n")
            f.write("-" * 50 + "\n")
    
    # Save grid map
    np.savetxt(f"{output_dir}/{base_name}_grid_{timestamp}.csv", grid_map, delimiter=',', fmt='%d')
    
    # Save route description to a separate text file
    desc_path = f"{output_dir}/{base_name}_description_{timestamp}.txt"
    with open(desc_path, "w") as f:
        f.write(route_description)
    
    # Create visualization
    complexity_metrics = analyze_route_complexity(grid_map)
    fig = visualize_grid_map(grid_map,
                             title=f"Route Grid Map{f' (Difficulty: {predicted_difficulty})' if predicted_difficulty else ''}")
    plt.figtext(0.5, 0.01, f"Density: {complexity_metrics['density']:.2f}, Holds: {complexity_metrics['num_holds']}", 
                ha="center", fontsize=9)
    fig.savefig(f"{output_dir}/{base_name}_route_map_{timestamp}.png")
    plt.close(fig)
    
    print(f"Results saved to {output_dir}/ folder")
    return txt_path, desc_path



import os
import sys
from typing import Optional

def setup_local_llm(model_name: str = "TheBloke/Llama-2-7B-Chat-GGUF") -> bool:
    """
    Check if the local LLM is installed, and if not, guide the user through installation.
    
    Args:
        model_name: Name of the LLM model to use (default: Llama-2-7B-Chat-GGUF)
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Check if llama-cpp-python is installed
        from llama_cpp import Llama
        print("Local LLM (llama-cpp-python) is already installed.")
    except ImportError:
        print("\nLocal LLM (llama-cpp-python) is not installed.")
        install = input("Would you like to install it now? (y/n): ").strip().lower()
        if install == 'y':
            print("\nInstalling llama-cpp-python...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
                print("Installation successful!")
                
                # Try importing again after installation
                try:
                    from llama_cpp import Llama
                except ImportError:
                    print("Installation appeared to succeed but module still can't be imported.")
                    print("Try restarting your Python environment or installing manually.")
                    return False
            except Exception as e:
                print(f"Error during installation: {e}")
                return False
        else:
            print("\nYou can install it later with 'pip install llama-cpp-python'")
            return False
    
    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("\nhuggingface_hub is not installed.")
        install = input("Would you like to install it now? (y/n): ").strip().lower()
        if install == 'y':
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            from huggingface_hub import hf_hub_download, list_repo_files
        else:
            print("\nYou can install it later with 'pip install huggingface_hub'")
            return False
    
    # List available GGUF files in the repository
    print(f"\nListing available GGUF files in {model_name}...")
    try:
        files = list_repo_files(model_name)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        if not gguf_files:
            print(f"No GGUF files found in {model_name}")
            return False
        
        # Sort by quantization level for better presentation
        gguf_files.sort()
        
        print("Available models:")
        for i, f in enumerate(gguf_files):
            print(f"{i+1}. {f}")
        
        # Let user select a model file
        selection = input(f"Select a model (1-{len(gguf_files)}) or press Enter for default (Q4_K_M): ")
        
        if selection.strip():
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(gguf_files):
                    selected_file = gguf_files[idx]
                else:
                    print("Invalid selection, using default model")
                    # Try to find a Q4_K_M model as default, fallback to first file
                    q4_models = [f for f in gguf_files if "Q4_K_M" in f]
                    selected_file = q4_models[0] if q4_models else gguf_files[0]
            except ValueError:
                print("Invalid input, using default model")
                # Try to find a Q4_K_M model as default, fallback to first file
                q4_models = [f for f in gguf_files if "Q4_K_M" in f]
                selected_file = q4_models[0] if q4_models else gguf_files[0]
        else:
            # Default to Q4_K_M model if available
            q4_models = [f for f in gguf_files if "Q4_K_M" in f]
            selected_file = q4_models[0] if q4_models else gguf_files[0]
        
        print(f"\nSelected model: {selected_file}")
        
        # Download the model and get its path
        print(f"Downloading/accessing model... (this may take a while)")
        model_path = hf_hub_download(repo_id=model_name, filename=selected_file)
        print(f"Model is available at: {model_path}")
        
        # Verify the model by loading it
        print("Testing model loading...")
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=model_path, n_ctx=2048)
            print("Model loaded successfully!")
            
            # Add a brief test
            test_prompt = "Q: Is NYU Tandon any good? A:"
            print(f"\nRunning quick test with prompt: '{test_prompt}'")
            response = llm(test_prompt, max_tokens=32, stop=["Q:", "\n"], echo=True)
            print("Response:", response["choices"][0]["text"])
            
            print("\nYou can use this model with the following path:")
            print(f"model_path = '{model_path}'")
            print("Example usage:")
            print("from llama_cpp import Llama")
            print(f"llm = Llama(model_path='{model_path}')")
            print("response = llm('Your prompt here', max_tokens=100)")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"The model file is at {model_path}")
            print("You can try loading it manually.")
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
'''
def generate_route_description(grid_map: np.ndarray, difficulty: Optional[str] = None, 
                              use_local_llm: bool = True, api_url: Optional[str] = None) -> str:
    """
    Generate a natural language description of the climbing route based on the grid map.
    
    Args:
        grid_map: 12x12 numpy array where 0=no hold, 1=small hold, 2=large hold/volume
        difficulty: Predicted difficulty level of the route
        use_local_llm: Whether to use a local LLM or an API
        api_url: URL for API-based LLM if not using local LLM
        
    Returns:
        str: Natural language description of the climbing route
    """
    # Convert grid to string representation for the prompt
    grid_str = ""
    for row in grid_map:
        grid_str += "".join(map(str, row)) + "\n"
    
    # Create a prompt for the LLM
    prompt = f"""
You are a professional climbing route setter. Analyze this 12x12 grid representing a climbing wall.
In this grid:
- 0 represents empty space (no holds)
- 1 represents a small hold
- 2 represents a large hold or volume (takes up two adjacent grid cells)

The bottom of the grid is the start, and the top is the end of the route.
Here is the grid map:

{grid_str}

{"The predicted difficulty is " + difficulty if difficulty else ""}

Provide a concise but informative description of this climbing route. Include:
1. The overall flow/pattern of the route
2. Any notable features (like long reaches, crux sections, rest positions)
3. The approximate difficulty based on hold density and positioning
4. Any recommendations for climbers attempting this route

Keep your response under 150 words and focus on being practical and helpful.
"""
    if use_local_llm:
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            
            # Get the correct path to your model using huggingface_hub
            try:
                # Use the exact model name and filename you downloaded
                model_path = hf_hub_download(
                    repo_id="TheBloke/Llama-2-7B-Chat-GGUF", 
                    filename="llama-2-7b-chat.Q4_K_M.gguf"  # Use the correct filename you downloaded
                )
                
                print(f"Using model from: {model_path}")
                
                # Initialize the LLM
                llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4
                )
                
                # Generate response
                response = llm(
                    prompt,
                    max_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    echo=False
                )
                
                return response['choices'][0]['text'].strip()
                
            except Exception as e:
                print(f"Error loading or using model: {e}")
                print("Falling back to default generic description...")
                return generate_generic_description(grid_map, difficulty)
        
        except ImportError as e:
            print(f"Required libraries not installed: {e}")
            print("Falling back to default generic description...")
            return generate_generic_description(grid_map, difficulty)
    
    elif api_url:
        try:
            # Using an API-based LLM service
            headers = {"Content-Type": "application/json"}
            payload = {
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                return response.json()["text"].strip()
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return generate_generic_description(grid_map, difficulty)
                
        except Exception as e:
            print(f"Error using API LLM: {e}")
            return generate_generic_description(grid_map, difficulty)
    
    else:
        return generate_generic_description(grid_map, difficulty)    
'''
def generate_route_description(grid_map: np.ndarray, difficulty: Optional[str] = None, 
                              use_local_llm: bool = True, api_url: Optional[str] = None) -> str:
    """
    Generate a natural language description of the climbing route based on the grid map.
    
    Args:
        grid_map: 12x12 numpy array where 0=no hold, 1=small hold, 2=large hold/volume
        difficulty: Predicted difficulty level of the route
        use_local_llm: Whether to use a local LLM or an API
        api_url: URL for API-based LLM if not using local LLM
        
    Returns:    
        str: Natural language description of the climbing route
    """
    # Convert grid to string representation for the prompt
    grid_str = ""
    for row in grid_map:
        grid_str += "".join(map(str, row)) + "\n"
    
    # Create a simplified prompt for the LLM
    prompt = f"""[INST]
You are a professional climbing route setter. Analyze this 12x12 grid representing a climbing wall.
In this grid:
- 0 represents empty space (no holds)
- 1 represents a small hold
- 2 represents a large hold 

The bottom of the grid is the start, and the top is the end of the route.
Here is the grid map:

{grid_str}

{"The predicted difficulty is " + difficulty if difficulty else ""}

Provide a concise but informative description of this climbing route (Don't introduce yourself and focus on being concise and providing maximal value to someone who cannot see the route
). Include:
1. The overall flow/pattern of the route
2. A simple but direct, bottom to top, description of how to get from one hold to the next (Do not include hold name).

Keep your response under 200 words and focus on being practical and helpful. [/INST]
"""
    if use_local_llm:
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            
            # Get the correct path to your model
            try:
                # Use the exact model name and filename you downloaded
                model_path = hf_hub_download(
                    repo_id="TheBloke/Llama-2-7B-Chat-GGUF", 
                    filename="llama-2-7b-chat.Q4_K_M.gguf"
                )
                
                print(f"Using model from: {model_path}")
                
                # Initialize the LLM with increased parameters
                llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,        # Increased context window
                    n_threads=4,
                    verbose=True       # Add verbose output for debugging
                )
                
                # Generate response with adjusted parameters
                response = llm(
                    prompt,
                    max_tokens=512,    # Increased token limit
                    temperature=0.0,   # Lower temperature for more deterministic output
                    top_p=0.95,
                    repeat_penalty=1.6, # Penalize repetitions
                    echo=False
                )
                
                # Debug output
                print("Raw response:", response)
                
                result_text = response['choices'][0]['text'].strip()
                if not result_text:
                    print("Empty response received, trying again with different settings...")
                    # Try again with different settings as a fallback
                    response = llm(
                        prompt,
                        max_tokens=256,
                        temperature=0.0,  # Zero temperature for most likely output
                        top_p=1.0,
                        echo=False
                    )
                    result_text = response['choices'][0]['text'].strip()
                
                return result_text if result_text else generate_generic_description(grid_map, difficulty)
                
            except Exception as e:
                print(f"Error loading or using model: {e}")
                print("Falling back to default generic description...")
                return generate_generic_description(grid_map, difficulty)
        
        except ImportError as e:
            print(f"Required libraries not installed: {e}")
            print("Falling back to default generic description...")
            return generate_generic_description(grid_map, difficulty)
    
    elif api_url:
        # API implementation remains the same
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                return response.json()["text"].strip()
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return generate_generic_description(grid_map, difficulty)
                
        except Exception as e:
            print(f"Error using API LLM: {e}")
            return generate_generic_description(grid_map, difficulty)
    
    else:
        return generate_generic_description(grid_map, difficulty)

def generate_generic_description(grid_map: np.ndarray, difficulty: Optional[str] = None) -> str:
    """
    Generate a generic route description based on simple analysis if LLM is not available.
    
    Args:
        grid_map: 12x12 numpy array of the climbing route
        difficulty: Predicted difficulty level
        
    Returns:
        str: Generic description of the route
    """
    # Count holds
    num_small_holds = np.sum(grid_map == 1)
    num_large_holds = np.sum(grid_map == 2) // 2  # Divide by 2 since each large hold takes 2 cells
    total_holds = num_small_holds + num_large_holds
    
    # Analyze hold distribution
    left_side = np.sum(grid_map[:, :6] > 0)
    right_side = np.sum(grid_map[:, 6:] > 0)
    
    top_section = np.sum(grid_map[:4, :] > 0)
    middle_section = np.sum(grid_map[4:8, :] > 0)
    bottom_section = np.sum(grid_map[8:, :] > 0)
    
    # Generate description
    description = f"This route contains {total_holds} holds ({num_small_holds} small, {num_large_holds} large). "
    
    if left_side > right_side * 1.5:
        description += "The route favors the left side of the wall. "
    elif right_side > left_side * 1.5:
        description += "The route favors the right side of the wall. "
    else:
        description += "The route is well balanced between left and right sides. "
    
    if bottom_section > middle_section and bottom_section > top_section:
        description += "The route has more holds at the bottom, suggesting a difficult start. "
    elif top_section > middle_section and top_section > bottom_section:
        description += "The route has more holds at the top, suggesting a challenging finish. "
    elif middle_section > bottom_section and middle_section > top_section:
        description += "The crux of the route appears to be in the middle section. "
    
    if difficulty:
        description += f"The estimated difficulty is {difficulty}. "
    
    return description

def analyze_route_complexity(grid_map: np.ndarray) -> Dict:
    """
    Analyze the complexity of a route based on its grid map.
    
    Args:
        grid_map: 12x12 numpy array of the climbing route
        
    Returns:
        Dict: Dictionary with complexity metrics
    """
    # Calculate density in different regions
    density = np.sum(grid_map > 0) / grid_map.size
    
    # Calculate average distance between holds
    hold_positions = np.argwhere(grid_map > 0)
    distances = []
    for i in range(len(hold_positions)):
        for j in range(i+1, len(hold_positions)):
            distances.append(np.linalg.norm(hold_positions[i] - hold_positions[j]))
    
    avg_distance = np.mean(distances) if distances else 0
    
    # Look for sequences that might be difficult
    rows, cols = grid_map.shape
    max_gap = 0
    for r in range(rows-1):
        if np.sum(grid_map[r,:]) > 0 and np.sum(grid_map[r+1,:]) > 0:
            continue
        gap = 1
        for r2 in range(r+2, rows):
            if np.sum(grid_map[r2,:]) > 0:
                break
            gap += 1
        max_gap = max(max_gap, gap)
    
    return {
        "density": density,
        "avg_distance": avg_distance,
        "max_vertical_gap": max_gap,
        "num_holds": np.sum(grid_map > 0),
        "num_volumes": np.sum(grid_map == 2) // 2
    }

def main():
    output_dir = "computer-vision/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("=" * 50)
    print("CLIMBING ROUTE ANALYZER (YOLO + Color Check + Grid Mapping + LLM Description)")
    print("=" * 50)
    
    # Image path input
    default_img = "computer-vision/climbing_wall.jpg"
    image_path = input(f"Enter path to image (default: {default_img}): ").strip()
    if not image_path:
        image_path = default_img
    
    # Color selection
    color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white",'pink']
    print("\nAvailable colors:", ", ".join(color_options))
    target_color = input("Enter color to detect (default: red): ").strip().lower()
    if not target_color or target_color not in color_options:
        target_color = "purple"
    
    # Detection parameters
    sensitivity_str = input("Enter sensitivity level (0-100, leave blank for auto): ").strip()

    if not sensitivity_str.isdigit():
        # Load image and calculate average brightness
        temp_img = cv2.imread(image_path)
        hsv_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2HSV)
        brightness = hsv_img[:, :, 2].mean()
        sensitivity = int(60 - (brightness / 255) * 40)
        sensitivity = max(10, min(60, sensitivity))
        print(f"[AUTO] Sensitivity set to {sensitivity} based on brightness")
    else:
        sensitivity = int(sensitivity_str)
        sensitivity = max(0, min(100, sensitivity))
        print(f"[MANUAL] Sensitivity set to {sensitivity}")

    
    min_area_str = input("Enter minimum area to consider as a hold (default: 200): ").strip()
    min_area = int(min_area_str) if min_area_str.isdigit() else 200
    
    # YOLO model path
    print("\nEnter YOLO model path (default: computer-vision/train4/weights/best.pt):")
    yolo_model_path = input().strip()
    if not yolo_model_path:
        yolo_model_path = "computer-vision/train4/weights/best.pt"
    
    # Difficulty prediction model
    use_model = input("Use difficulty prediction model? (y/n, default: y): ").strip().lower() != 'n'
    model = None
    if use_model:
        model_path = input("Enter classification model path (leave blank for demo): ").strip()
        if not model_path:
            model_path = None
        model = load_model(model_path)
    
    # LLM for route description
    use_llm = input("Generate route description using LLM? (y/n, default: y): ").strip().lower() != 'n'
    llm_api_url = None
    
    if use_llm:
        llm_type = input("Use local LLM or API? (local/api, default: local): ").strip().lower()
        if llm_type == "api":
            llm_api_url = input("Enter API URL for LLM service: ").strip()
        else:
            # Check and setup local LLM
            setup_success = setup_local_llm()
            if not setup_success:
                print("Warning: Local LLM setup was not completed. Will use generic description.")
    
    print("\nProcessing image with YOLO detection + color check...")
    holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
        image_path,
        target_color=target_color,
        sensitivity=sensitivity,
        min_area=min_area,
        yolo_model_path=yolo_model_path
    )
    
    predicted_difficulty = None
    if model and len(holds_info) > 0:
        predicted_difficulty = predict_route_difficulty(model, grid_map)
        print(f"\nPredicted route difficulty: {predicted_difficulty}")
    
    print(f"\nDetected {len(holds_info)} holds:")
    for i, hold in enumerate(holds_info):
        print(f"Hold {i+1}: {hold['type']} at grid {hold['grid_position']} (Points: {hold['points_value']})")
    
    # Visualize results
    try:
        original_image = cv2.imread(image_path)
        crop_x, crop_y, crop_w, crop_h = cropped_region
        cropped_image = original_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 2)
        plt.title("Masked Image")
        plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 3)
        plt.title("Result Image (BBoxes)")
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 4)
        plt.title("Cropped Region")
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 5)
        plt.title("Grid Map")
        plt.imshow(grid_map, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        
        plt.subplot(2, 3, 6)
        plt.title("Route Difficulty")
        plt.text(0.5, 0.5, predicted_difficulty if predicted_difficulty else "N/A",
                 horizontalalignment='center', verticalalignment='center', fontsize=18)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying images: {e}")
    
    # Save all results and generate route description
    txt_path, desc_path = save_results(
        image_path, holds_info, grid_map, result_image, masked_image, cropped_region,
        predicted_difficulty, output_dir, use_llm, llm_api_url
    )
    
    # Ask if user wants to read the description aloud
    read_aloud = input("\nRead route description aloud? (y/n, default: n): ").strip().lower() == 'y'
    if read_aloud:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            with open(desc_path, 'r') as f:
                description = f.read()
                
            print("Reading route description...")
            engine.say(description)
            engine.runAndWait()
        except ImportError:
            print("Text-to-speech module (pyttsx3) not installed.")
            install_tts = input("Install pyttsx3 now? (y/n): ").strip().lower()
            if install_tts == 'y':
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyttsx3"])
                    print("Please run the program again to use text-to-speech.")
                except Exception as e:
                    print(f"Error installing pyttsx3: {e}")
        except Exception as e:
            print(f"Error reading description aloud: {e}")
            
if __name__ == "__main__":
    main()
