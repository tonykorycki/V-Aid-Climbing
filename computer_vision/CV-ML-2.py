import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch
from ultralytics import YOLO
from PIL import Image
from io import BytesIO 

# "python -m venv venv" to open virtual machine. ".venv\Scripts\activate" to activate virtual machine. "deactivate" to deactivate virtual machine.
# pip install opencv-python numpy matplotlib torch ultralytics pillow (Install these libraries if you're trying to run this file); 

#This file is a combination of the previous two files. It uses YOLO to detect holds and volumes, and then uses a color check to determine if the hold is valid.
#It then maps the holds onto a 12x12 grid and predicts the difficulty of the route based on the grid map.

##############################################################################
# YOLO + COLOR-BASED HOLD/Volume DETECTION WITH GRID MAPPING
##############################################################################

def detect_and_classify_holds(image_path,
                              target_color='red',
                              sensitivity=25,
                              min_area=30,
                              yolo_model_path="computer_vision/train4/weights/best.pt"):
    """
    Uses a YOLO model to detect objects. For each detected bounding box:
      - If the predicted class is “volume” (or if it’s a hold but very large), then we mark it as volume.
      - For non-volume holds, we perform a color check (using your HSV ranges) to decide if it’s valid.
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

    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return [], np.zeros((12, 12), dtype=np.int32), image, image, (0, 0, orig_w, orig_h)

    # Run YOLO on the image (adjust confidence threshold if needed)
    results = yolo_model.predict(source=image, conf=0.25)
    if len(results) == 0 or len(results[0].boxes) == 0:
        print("No holds detected by YOLO.")
        return [], np.zeros((12, 12), dtype=np.int32), image, image, (0, 0, orig_w, orig_h)

    # Define HSV color ranges (same as your demo code)
    
    color_ranges = {
        'red':   (np.array([0, 150, 80]),   np.array([6, 255, 255])),      # Narrow red to avoid orange
        'red2':  (np.array([174, 150, 80]), np.array([180, 255, 255])),
        'blue':  (np.array([100, 150, 50]), np.array([120, 255, 255])),    # Narrow blue to avoid purple
        'green': (np.array([40, 50, 40]),   np.array([80, 255, 255])),
        'yellow':(np.array([25, 150, 150]), np.array([35, 255, 255])),     # Tighten yellow slightly
        'orange':(np.array([10, 150, 150]), np.array([16, 255, 255])),     # Narrow orange to avoid yellow overlap
        'purple':(np.array([120, 40, 40]),  np.array([150, 255, 255])),     # As before – blue now is separate
        'black': (np.array([0, 0, 0]),      np.array([180, 60, 40])),
        'white': (np.array([0, 0, 200]),    np.array([180, 50, 255]))       # Lower the V threshold slightly to catch brighter whites
    }
    """
    color_ranges = {
    
        'red':   (np.array([0, 150, 80]), np.array([10 + sensitivity // 5, 255, 255])),
        'red2':  (np.array([170 - sensitivity // 5, 120, 70]), np.array([180, 255, 255])),
        'blue':  (np.array([120 - sensitivity // 5, 150, 50]), np.array([110 + sensitivity // 5, 255, 255])),
        'green': (np.array([45 - sensitivity // 5, 40, 30]), np.array([70 + sensitivity // 5, 255, 255])),
        'yellow':(np.array([30 - sensitivity // 5, 150, 150]), np.array([30 + sensitivity // 5, 255, 255])),
        'orange':(np.array([15 - sensitivity // 5, 150, 150]), np.array([13 + sensitivity // 5, 255, 255])),
        'purple':(np.array([125 - sensitivity // 5, 40, 40]), np.array([165 + sensitivity // 5, 255, 255])),
        'black': (np.array([0, 0, 0]), np.array([180, 60, 35 + sensitivity // 2])),
        'white': (np.array([0, 0, 200 - sensitivity // 5]), np.array([180, 45 + sensitivity // 2, 255]))
    }
    """
    


    
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
        # Place points based on the hold’s shape using the bounding box
        bbox = hold["bbox"]
        fill_grid_points_by_shape(grid_map, grid_x, grid_y, hold["points_value"],
                                  bbox, (crop_x, crop_y, crop_w, crop_h),
                                  cell_width, cell_height)
        hold["relative_position"] = (rel_x, rel_y)
        hold["grid_position"] = (grid_x, grid_y)
    # Filter out volumes that are too isolated

    
    volume_threshold = 0.05 * min(orig_w, orig_h)  # e.g., 10% of the smaller image dimension
    holds_info = filter_isolated_volumes(holds_info, volume_threshold)

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
                 predicted_difficulty=None, output_dir="computer-vison/results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"{output_dir}/{base_name}_detected_{timestamp}.jpg", result_image)
    cv2.imwrite(f"{output_dir}/{base_name}_masked_{timestamp}.jpg", masked_image)
    x, y, w, h = cropped_region
    orig_img = cv2.imread(image_path)
    cropped_img = orig_img[y:y+h, x:x+w]
    cv2.imwrite(f"{output_dir}/{base_name}_cropped_{timestamp}.jpg", cropped_img)
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
            f.write(f"  Relative position in crop: {hold['relative_position']}\n")
            f.write(f"  Grid position: {hold['grid_position']}\n")
            f.write(f"  Dimensions: {hold['dimensions']}\n")
            f.write(f"  Area: {hold['area']} pixels\n")
            f.write(f"  Points: {hold.get('points_value', 1)}\n")
            f.write("-" * 50 + "\n")
    np.savetxt(f"{output_dir}/{base_name}_grid_{timestamp}.csv", grid_map, delimiter=',', fmt='%d')
    fig = visualize_grid_map(grid_map,
                             title=f"Route Grid Map{f' (Difficulty: {predicted_difficulty})' if predicted_difficulty else ''}")
    fig.savefig(f"{output_dir}/{base_name}_route_map_{timestamp}.png")
    plt.close(fig)
    print(f"Results saved to {output_dir}/ folder")
    return txt_path

def main():
    output_dir = "computer_vision/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("=" * 50)
    print("CLIMBING ROUTE ANALYZER (YOLO + Color Check + Grid Mapping)")
    print("=" * 50)
    default_img = "computer_vision/climbing_wall.jpg"
    image_path = input(f"Enter path to image (default: {default_img}): ").strip()
    if not image_path:
        image_path = default_img
    color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]
    print("\nAvailable colors:", ", ".join(color_options))
    target_color = input("Enter color to detect (default: red): ").strip().lower()
    if not target_color or target_color not in color_options:
        target_color = "red"
    sensitivity_str = input("Enter sensitivity level (0-100, default: 25): ").strip()
    sensitivity = int(sensitivity_str) if sensitivity_str.isdigit() else 25
    sensitivity = max(0, min(100, sensitivity))
    min_area_str = input("Enter minimum area to consider as a hold (default: 30): ").strip()
    min_area = int(min_area_str) if min_area_str.isdigit() else 30
    print("\nEnter YOLO model path (default: computer_vision/train4/weights/best.pt):")
    yolo_model_path = input().strip()
    if not yolo_model_path:
        yolo_model_path = "computer_vision/train4/weights/best.pt"
    use_model = input("Use difficulty prediction model? (y/n, default: y): ").strip().lower() != 'n'
    model = None
    if use_model:
        model_path = input("Enter classification model path (leave blank for demo): ").strip()
        if not model_path:
            model_path = None
        model = load_model(model_path)
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
    save_results(image_path, holds_info, grid_map, result_image, masked_image, cropped_region,
                 predicted_difficulty, output_dir)

if __name__ == "__main__":
    main()
