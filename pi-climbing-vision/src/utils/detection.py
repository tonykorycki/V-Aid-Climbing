from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import os

def detect_and_classify_holds(image_path: str,
                              yolo_model_path: str,
                              target_color: str = 'red',
                              sensitivity: int = 25,
                              min_area: int = 200) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Detect holds and volumes in the image using YOLO and color detection.

    Args:
        image_path (str): Path to the input image.
        yolo_model_path (str): Path to the YOLO model weights.
        target_color (str): Color of the holds to detect.
        sensitivity (int): Sensitivity for color detection.
        min_area (int): Minimum area to consider a detection valid.

    Returns:
        Tuple containing:
            - holds_info: List of dictionaries with detected hold information.
            - grid_map: 12x12 numpy array representing the grid mapping of holds.
            - masked_image: Image showing only the detected holds.
            - result_image: Image with bounding boxes drawn around detected holds.
            - cropped_region: Tuple representing the bounding box of the detected region (x, y, width, height).
    """
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    orig_h, orig_w = image.shape[:2]

    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)

    # Run YOLO on the image
    results = yolo_model.predict(source=image, conf=0.25)
    if len(results) == 0 or len(results[0].boxes) == 0:
        print("No holds detected by YOLO.")
        return [], np.zeros((12, 12), dtype=np.int32), image, image, (0, 0, orig_w, orig_h)

    # Define HSV color ranges
    color_ranges = {
        'red': (np.array([0, 150, 80]), np.array([6, 255, 255])),
        'red2': (np.array([174, 150, 80]), np.array([180, 255, 255])),
        'blue': (np.array([100, 150, 50]), np.array([120, 255, 255])),
        'green': (np.array([40, 50, 40]), np.array([80, 255, 255])),
        'yellow': (np.array([25, 150, 150]), np.array([35, 255, 255])),
        'orange': (np.array([10, 150, 150]), np.array([16, 255, 255])),
        'purple': (np.array([120, 40, 40]), np.array([150, 255, 255])),
        'black': (np.array([0, 0, 0]), np.array([180, 60, 40])),
        'white': (np.array([0, 0, 200]), np.array([180, 50, 255]))
    }

    def is_target_color(roi_bgr):
        """Check if the ROI contains the target color."""
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lower_bound, upper_bound = color_ranges[target_color]
        mask = cv2.inRange(roi_hsv, lower_bound, upper_bound)
        return (mask > 0).mean() > 0.2

    holds_info = []
    masked_image = np.zeros_like(image)

    # Loop over YOLO bounding boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_area = (x2 - x1) * (y2 - y1)

        if box_area < min_area:
            continue

        # Get predicted class from YOLO
        pred_class = int(box.cls[0].item()) if hasattr(box, 'cls') else 0

        # Perform color check for non-volume holds
        if pred_class == 0:  # Assuming 0 is the class for holds
            roi = image[y1:y2, x1:x2]
            color_pass = is_target_color(roi)
            if not color_pass:
                continue

        # Record hold information
        holds_info.append({
            "type": "volume" if pred_class == 1 else "hold",
            "position": ((x1 + x2) // 2, (y1 + y2) // 2),
            "bbox": (x1, y1, x2, y2),
            "area": box_area
        })
        masked_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    # Create a 12x12 grid map
    grid_map = np.zeros((12, 12), dtype=np.int32)

    # Map holds to grid
    for hold in holds_info:
        cx, cy = hold["position"]
        grid_x = min(11, max(0, cx * 12 // orig_w))
        grid_y = min(11, max(0, cy * 12 // orig_h))
        grid_map[grid_y, grid_x] = 1  # Mark as hold

    # Create result_image (with bounding boxes)
    result_image = image.copy()
    for hold in holds_info:
        x1, y1, x2, y2 = hold["bbox"]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Determine crop region
    min_x = min([hold["bbox"][0] for hold in holds_info]) if holds_info else 0
    min_y = min([hold["bbox"][1] for hold in holds_info]) if holds_info else 0
    max_x = max([hold["bbox"][2] for hold in holds_info]) if holds_info else orig_w
    max_y = max([hold["bbox"][3] for hold in holds_info]) if holds_info else orig_h
    cropped_region = (min_x, min_y, max_x-min_x, max_y-min_y)
    
    return holds_info, grid_map, masked_image, result_image, cropped_region

def fetch_llm_description(grid_map: np.ndarray, api_url: str) -> str:
    """
    Fetch a description from the LLM API based on the grid map.

    Args:
        grid_map (np.ndarray): 12x12 grid map of holds.
        api_url (str): URL of the LLM API.

    Returns:
        str: Generated description from the LLM.
    """
    import requests
    import json

    grid_str = "\n".join("".join(map(str, row)) for row in grid_map)
    prompt = f"Analyze this 12x12 grid representing a climbing wall:\n{grid_str}"

    response = requests.post(api_url, json={"prompt": prompt})
    if response.status_code == 200:
        return response.json().get("text", "No description generated.")
    else:
        raise Exception(f"Error fetching description: {response.status_code} - {response.text}")