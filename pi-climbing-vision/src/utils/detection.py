from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_wall_area(image):
    """
    Detect the climbing wall area in the image to isolate it from the background.
    
    Args:
        image: Input image
        
    Returns:
        Tuple containing:
            - mask: Binary mask of the wall area
            - region: (x, y, w, h) coordinates of the wall area
    """
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
                              yolo_model_path=None):
    """
    Uses a YOLO model to detect and classify climbing holds in an image.
    
    Args:
        image_path: Path to the input image.
        target_color: Color of holds to detect (red, blue, etc.)
        sensitivity: Sensitivity for color detection.
        min_area: Minimum area to consider a detection valid.
        yolo_model_path: Path to the YOLO model weights.
        
    Returns:
        Tuple containing:
            - holds_info: List of dictionaries with hold information.
            - grid_map: 12x12 numpy array representing the grid mapping.
            - masked_image: Image showing only detected holds.
            - result_image: Image with annotations.
            - cropped_region: (x, y, w, h) of the region of interest.
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

    # Run YOLO on the image
    results = yolo_model.predict(source=wall_image, conf=0.25)
    if len(results) == 0 or len(results[0].boxes) == 0:
        print("No holds detected by YOLO.")
        return [], np.zeros((12, 12), dtype=np.int32), image, image, (0, 0, orig_w, orig_h)

    # Define HSV color ranges with adaptive sensitivity
    v_base = 50 - sensitivity // 4  # more forgiving with brightness
    s_base = 100 - sensitivity // 4

    color_ranges = {
        'red':   (np.array([0, 150, 80]),   np.array([6, 255, 255])),
        'red2':  (np.array([174, 150, 80]), np.array([180, 255, 255])),
        'blue':  (np.array([90, s_base, v_base]), np.array([130, 255, 255])),
        'green': (np.array([40, 50, 40]),   np.array([80, 255, 255])),
        'yellow':(np.array([25, 150, 150]), np.array([35, 255, 255])),
        'orange':(np.array([10, 150, 150]), np.array([16, 255, 255])),
        'purple':(np.array([120, 40, 40]),  np.array([150, 255, 255])),
        'black': (np.array([0, 0, 0]),      np.array([180, 60, 40])),
        'white': (np.array([0, 0, 200]),    np.array([180, 50, 255])),
        'pink':  (np.array([160, 80, 80]),  np.array([175, 255, 255]))
    }

    def is_target_color(roi_bgr):
        """Check if the ROI contains the target color."""
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
        
        # Adjust coordinates to account for wall cropping
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

        # Get predicted class from YOLO
        pred_class = int(box.cls[0].item()) if hasattr(box, 'cls') else 0

        # Calculate bounding box area
        box_area = (x2 - x1) * (y2 - y1)

        # Dismiss a volume if its area is more than 15% of the total image area
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
            "bbox": (x1, y1, x2, y2),
            "relative_position": None,  # to be filled after cropping
            "grid_position": None,      # to be filled during grid mapping
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

    # Import grid mapping functions
    from utils.grid_mapping import fill_grid_points_by_shape, filter_isolated_volumes

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
    volume_threshold = 0.05 * min(orig_w, orig_h)  # e.g., 5% of the smaller image dimension
    filtered_holds = filter_isolated_volumes(holds_info, volume_threshold)
    
    return filtered_holds, grid_map, masked_image, result_image, cropped_region

def analyze_route_complexity(grid_map):
    """
    Analyze the complexity of a climbing route based on its grid map.
    
    Args:
        grid_map: 12x12 numpy array representing the route
        
    Returns:
        Dictionary with complexity metrics
    """
    # Calculate density
    density = np.sum(grid_map > 0) / grid_map.size
    
    # Calculate average distance between holds
    hold_positions = np.argwhere(grid_map > 0)
    distances = []
    for i in range(len(hold_positions)):
        for j in range(i+1, len(hold_positions)):
            distances.append(np.linalg.norm(hold_positions[i] - hold_positions[j]))
    
    avg_distance = np.mean(distances) if distances else 0
    
    # Look for vertical gaps
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