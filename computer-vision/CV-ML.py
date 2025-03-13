import cv2
print(cv2.__version__)
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch  # For model integration
import requests
from PIL import Image
from io import BytesIO

def detect_and_classify_holds(image_path, target_color='red', sensitivity=25, min_area=200):
    """
    Detect climbing holds of a specified color and classify them
    
    Args:
        image_path (str): Path to the input image
        target_color (str): Color to detect (red, blue, green, yellow, etc.)
        sensitivity (int): Color detection sensitivity (0-100, higher = more lenient)
        min_area (int): Minimum contour area to consider as a hold
        
    Returns:
        tuple: (holds_info, grid_map, masked_image, result_image, cropped_region)
    """
    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Improved color ranges based on common climbing hold colors
    # The values are in HSV (Hue, Saturation, Value) format
    color_ranges = {
        'red': (np.array([0, 120, 70]), np.array([10 + sensitivity//5, 255, 255])),  # Lower red range
        'red2': (np.array([170 - sensitivity//5, 120, 70]), np.array([180, 255, 255])),  # Upper red range
        'blue': (np.array([100 - sensitivity//5, 100, 50]), np.array([130 + sensitivity//5, 255, 255])),
        'green': (np.array([35 - sensitivity//5, 40, 30]), np.array([90 + sensitivity//5, 255, 255])),
        'yellow': (np.array([20 - sensitivity//5, 100, 100]), np.array([35 + sensitivity//5, 255, 255])),
        'orange': (np.array([10 - sensitivity//5, 100, 100]), np.array([25 + sensitivity//5, 255, 255])),
        'purple': (np.array([125 - sensitivity//5, 40, 40]), np.array([165 + sensitivity//5, 255, 255])),
        'black': (np.array([0, 0, 0]), np.array([180, 70, 40 + sensitivity//2])),
        'white': (np.array([0, 0, 180 - sensitivity*2]), np.array([180, 40 + sensitivity//2, 255]))
    }
    
    # Handle special case for red (which wraps around the hue spectrum)
    if target_color == 'red':
        lower_bound1, upper_bound1 = color_ranges['red']
        lower_bound2, upper_bound2 = color_ranges['red2']
        
        mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif target_color in color_ranges:
        lower_bound, upper_bound = color_ranges[target_color]
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
    else:
        raise ValueError(f"Unsupported color: {target_color}. Available colors: {list(color_ranges.keys())}")
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to get only the target color
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Find contours of holds
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found:", len(contours))
    for idx, cnt in enumerate(contours):
        print(f"Contour {idx}: type {type(cnt)}", "shape:", cnt.shape if hasattr(cnt, "shape") else "N/A")



    
    # Create a copy for visualization
    result_image = image.copy()
    
    # Filter contours by minimum area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    
    # If no valid contours found
    if not valid_contours:
        print(f"No valid holds detected with current settings.")
        cropped_region = (0, 0, image.shape[1], image.shape[0])  # Full image
        holds_info = []
        grid_map = np.zeros((12, 12), dtype=np.int32)  # Modified to 12x12
        return holds_info, grid_map, masked_image, result_image, cropped_region
    
    # Find bounding region of all detected holds for cropping
    all_points = np.vstack([cnt for cnt in valid_contours])
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add padding around the holds region (10% padding)
    padding_x = int(w * 0.1)
    padding_y = int(h * 0.1)
    
    # Ensure we don't go out of bounds
    crop_x = max(0, x - padding_x)
    crop_y = max(0, y - padding_y)
    crop_w = min(image.shape[1] - crop_x, w + padding_x * 2)
    crop_h = min(image.shape[0] - crop_y, h + padding_y * 2)
    
    # Define the crop region
    cropped_region = (crop_x, crop_y, crop_w, crop_h)
    
    # Create a 12x12 grid map (changed from 40x26)
    grid_map = np.zeros((12, 12), dtype=np.int32)
    
    # Calculate cell dimensions based on the cropped region
    cell_width = crop_w / 12  # 12 columns
    cell_height = crop_h / 12  # 12 rows
    
    # Process each contour (potential hold)
    holds_info = []
    for cnt in valid_contours:
        area = cv2.contourArea(cnt)
            
        # Get bounding rectangle
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        
        # Calculate relative position within the cropped region
        rel_x = x_cnt - crop_x
        rel_y = y_cnt - crop_y
        
        # Calculate center point
        center_x, center_y = x_cnt + w_cnt//2, y_cnt + h_cnt//2
        rel_center_x, rel_center_y = center_x - crop_x, center_y - crop_y
        
        # Map center to grid coordinates
        grid_x = max(0, min(11, int(rel_center_x / cell_width)))
        grid_y = max(0, min(11, int(rel_center_y / cell_height)))
        
        # Classify hold
        hold_type = classify_hold(cnt)
        
        # Determine points to fill based on hold size/type
        points_to_fill = determine_points_to_fill(cnt, hold_type, area)
        
        # Check if hold is within the cropped region
        if 0 <= rel_x < crop_w and 0 <= rel_y < crop_h:
            # Fill the grid map based on hold size
            fill_grid_points(grid_map, grid_x, grid_y, points_to_fill, rel_center_x, rel_center_y, 
                            cnt, rel_x, rel_y, cell_width, cell_height)
            
            # Store hold information
            holds_info.append({
                "type": hold_type,
                "position": (center_x, center_y),
                "relative_position": (rel_center_x, rel_center_y),
                "grid_position": (grid_x, grid_y),
                "dimensions": (w_cnt, h_cnt),
                "area": area,
                "points_value": points_to_fill  # Number of points this hold is worth
            })
            
            # Draw bounding box and label on result image
            cv2.rectangle(result_image, (x_cnt, y_cnt), (x_cnt+w_cnt, y_cnt+h_cnt), (0, 255, 0), 2)
            cv2.putText(result_image, f"{hold_type} ({points_to_fill})", (x_cnt, y_cnt - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            
            # Draw the grid position on the result image
            cv2.circle(result_image, (center_x, center_y), 3, (0, 255, 255), -1)
            cv2.putText(result_image, f"({grid_x},{grid_y})", (center_x + 5, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw the cropping rectangle on the result image
    cv2.rectangle(result_image, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 2)
    
    return holds_info, grid_map, masked_image, result_image, cropped_region

def determine_points_to_fill(contour, hold_type, area):
    """
    Determine how many points to fill on the grid based on hold size and type
    
    Args:
        contour: The contour of the hold
        hold_type: The classified type of the hold
        area: The area of the hold in pixels
        
    Returns:
        int: Number of points to fill (1, 2, or 3)
    """
    # Basic size thresholds (adjust based on your image resolution)
    small_threshold = 500
    large_threshold = 2000
    
    # Special case for volumes
    if hold_type.lower() == "volume":
        return 3
    
    # Size-based determination for other holds
    if area < small_threshold:
        return 1  # Small holds get 1 point
    elif area < large_threshold:
        return 2  # Medium holds get 2 points
    else:
        return 3  # Large holds (like jugs) get 3 points

def fill_grid_points(grid_map, center_x, center_y, points_to_fill, rel_center_x, rel_center_y, 
                    contour, rel_x, rel_y, cell_width, cell_height):
    """
    Fill points on the grid map to represent the hold
    
    Args:
        grid_map: The grid map to fill
        center_x, center_y: The center grid coordinates
        points_to_fill: Number of points to fill (1-3)
        rel_center_x, rel_center_y: Relative position in the cropped region
        contour: The hold contour
        rel_x, rel_y: Relative top-left corner position
        cell_width, cell_height: Grid cell dimensions
    """
    # Always fill the center point
    grid_map[center_y, center_x] = points_to_fill
    
    # If only 1 point to fill, we're done
    if points_to_fill == 1:
        return
    
    # For 2 or 3 points, we need to determine the hold shape
    # Create a mask of the contour
    mask = np.zeros((int(12 * cell_height), int(12 * cell_width)), dtype=np.uint8)
    adjusted_contour = contour.copy()
    adjusted_contour[:, :, 0] = adjusted_contour[:, :, 0] - int(rel_x)
    adjusted_contour[:, :, 1] = adjusted_contour[:, :, 1] - int(rel_y)
    cv2.drawContours(mask, [adjusted_contour], 0, 255, -1)
    
    # For 2 points: find one additional point in the direction of the main axis
    if points_to_fill == 2:
        # Find the major axis direction using PCA
        moments = cv2.moments(contour)
        if moments['mu20'] + moments['mu02'] != 0:  # Avoid division by zero
            # Get covariance matrix elements from moments
            a = moments['mu20'] / moments['m00']
            b = moments['mu11'] / moments['m00']
            c = moments['mu02'] / moments['m00']
            
            # Find angle of the major axis
            theta = 0.5 * np.arctan2(2*b, a-c)
            
            # Direction vector for major axis
            dx = np.cos(theta)
            dy = np.sin(theta)
            
            # Check cells in major axis direction
            potential_points = []
            for offset in [-1, 1]:  # Check both directions
                new_x = int(center_x + offset * dx)
                new_y = int(center_y + offset * dy)
                
                # Check if within grid and inside contour
                if (0 <= new_x < 12 and 0 <= new_y < 12):
                    # Check if this cell contains part of the hold
                    cell_center_x = int((new_x + 0.5) * cell_width)
                    cell_center_y = int((new_y + 0.5) * cell_height)
                    
                    if 0 <= cell_center_x < mask.shape[1] and 0 <= cell_center_y < mask.shape[0]:
                        if mask[cell_center_y, cell_center_x] > 0:
                            potential_points.append((new_x, new_y))
            
            # If we found any valid points, use the first one
            if potential_points:
                x, y = potential_points[0]
                grid_map[y, x] = points_to_fill
    
    # For 3 points: Add two more points to capture the shape
    elif points_to_fill == 3:
        # For 3 points, we want to add points around the center
        # to capture the general shape of the hold
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Count how many neighbors we've added
        added_neighbors = 0
        
        # Check all neighboring cells
        for dx, dy in neighbors:
            new_x = center_x + dx
            new_y = center_y + dy
            
            # Check if within grid
            if 0 <= new_x < 12 and 0 <= new_y < 12:
                # Check if this cell contains part of the hold
                cell_center_x = int((new_x + 0.5) * cell_width)
                cell_center_y = int((new_y + 0.5) * cell_height)
                
                if 0 <= cell_center_x < mask.shape[1] and 0 <= cell_center_y < mask.shape[0]:
                    if mask[cell_center_y, cell_center_x] > 0:
                        grid_map[new_y, new_x] = points_to_fill
                        added_neighbors += 1
                        
                        # Stop after adding 2 more points
                        if added_neighbors >= 2:
                            break

def classify_hold(cnt, area_adjustment_factor=1.0):
    """
    Classify hold based on shape features with adjustment for smaller holds
    
    Args:
        cnt: Contour of the potential hold
        area_adjustment_factor: Factor to adjust area thresholds (lower for smaller holds)
        
    Returns:
        str: Hold type classification
    """
    # Calculate shape features
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Aspect ratio (width/height)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h) if h != 0 else 0
    
    # Circularity (4*pi*area/perimeter^2)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
    
    # Convexity (higher = smoother)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0
    
    # Approximate polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Solidity (ratio of contour area to its convex hull area)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    
    # Adjusted area thresholds for smaller holds
    micro_threshold = 500 * area_adjustment_factor
    jug_threshold = 5000 * area_adjustment_factor
    
    # Extended classification based on multiple features
    if area < micro_threshold:
        return "micro"
    elif area > jug_threshold and convexity > 0.85:
        return "jug"
    elif aspect_ratio > 1.5 and solidity > 0.8:
        return "rail"
    elif circularity > 0.75 and convexity > 0.85:
        return "pinch"
    elif len(approx) > 6 and solidity < 0.85:
        return "sloper"
    elif aspect_ratio < 1.3 and convexity < 0.8:
        return "crimp"
    elif 0.7 < convexity < 0.9 and 0.65 < solidity < 0.9:
        return "pocket"
    else:
        return "foothold"

def load_model(model_path=None):
    """
    Load the climbing route model
    
    Args:
        model_path: Path to the model file, or None to download
        
    Returns:
        model: The loaded model
    """
    try:
        # Check if PyTorch is available
        import torch
        
        # If model_path is not provided, you can set default path or download from URL
        if model_path is None:
            # Either download from GitHub or set a default local path
            # Example: model_path = "model/climbingcrux_model.pth"
            print("No model path provided. Attempting to download model...")
            
            # For demo purposes, we'll use a placeholder
            # In real implementation, this would download the model from GitHub
            
            # Create a dummy model for demonstration
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
                    x = x.view(-1, 32 * 3 * 3)  # Flatten
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            model = ClimbingRouteModel()
            print("Demonstration model created (not actually loaded from file)")
            
        else:
            # Load model from file
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()  # Set to evaluation mode
            print(f"Model loaded from {model_path}")
        
        return model
        
    except ImportError:
        print("PyTorch is not installed. Cannot load model.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_route_difficulty(model, grid_map):
    """
    Predict route difficulty based on the grid map
    
    Args:
        model: The loaded model
        grid_map: The 12x12 grid map of holds
        
    Returns:
        difficulty: Predicted difficulty
    """
    try:
        import torch
        
        # Convert grid map to tensor
        input_tensor = torch.tensor(grid_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Map prediction to difficulty grades
        difficulty_grades = ["V0-V1", "V2-V3", "V4-V5", "V6-V7", "V8-V9", "V10+"]
        predicted_difficulty = difficulty_grades[predicted.item()]
        
        return predicted_difficulty
        
    except Exception as e:
        print(f"Error predicting route difficulty: {e}")
        return "Unknown"

def visualize_grid_map(grid_map, title="Route Grid Map"):
    """
    Visualize the 12x12 grid map
    
    Args:
        grid_map: The 12x12 grid map
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    
    # Create a colormap: 0=empty, 1=small, 2=medium, 3=large/volume
    cmap = plt.cm.Blues
    plt.imshow(grid_map, cmap=cmap, interpolation='nearest')
    
    # Add grid lines
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    for i in range(grid_map.shape[0] + 1):
        plt.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5)
    for i in range(grid_map.shape[1] + 1):
        plt.axvline(x=i-0.5, color='gray', linestyle='-', linewidth=0.5)
    
    # Add values in cells
    for i in range(grid_map.shape[0]):
        for j in range(grid_map.shape[1]):
            if grid_map[i, j] > 0:
                plt.text(j, i, str(grid_map[i, j]), 
                         ha="center", va="center", 
                         color="white" if grid_map[i, j] > 1 else "black")
    
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()  # Return the figure

def save_results(image_path, holds_info, grid_map, result_image, masked_image, cropped_region, 
                predicted_difficulty=None, output_dir="results"):
    """
    Save detection results to files
    
    Args:
        image_path (str): Path to the original image
        holds_info (list): List of detected holds information
        grid_map (numpy.ndarray): The binary grid map
        result_image: Annotated image with hold classifications
        masked_image: Image with only the target color visible
        cropped_region (tuple): (x, y, w, h) of the cropped region
        predicted_difficulty (str): Predicted route difficulty
        output_dir (str): Directory to save results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get base filename without extension
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save annotated image
    cv2.imwrite(f"{output_dir}/{base_name}_detected_{timestamp}.jpg", result_image)
    
    # Save masked image
    cv2.imwrite(f"{output_dir}/{base_name}_masked_{timestamp}.jpg", masked_image)
    
    # Extract and save the cropped region of the original image
    x, y, w, h = cropped_region
    cropped_image = cv2.imread(image_path)[y:y+h, x:x+w]
    cv2.imwrite(f"{output_dir}/{base_name}_cropped_{timestamp}.jpg", cropped_image)
    
    # Save holds information as text
    with open(f"{output_dir}/{base_name}_holds_{timestamp}.txt", "w") as f:
        f.write(f"Detected {len(holds_info)} holds:\n")
        f.write(f"Cropped region: x={x}, y={y}, w={w}, h={h}\n")
        if predicted_difficulty:
            f.write(f"Predicted difficulty: {predicted_difficulty}\n")
        f.write("-" * 50 + "\n")
        for i, hold in enumerate(holds_info):
            f.write(f"Hold {i+1}: {hold['type']}\n")
            f.write(f"  Position in original: {hold['position']}\n")
            f.write(f"  Position in crop: {hold['relative_position']}\n")
            f.write(f"  Grid position: {hold['grid_position']}\n")
            f.write(f"  Size: {hold['dimensions']}\n")
            f.write(f"  Area: {hold['area']} pixels\n")
            f.write(f"  Points: {hold.get('points_value', 1)}\n")
            f.write("-" * 50 + "\n")
    
    # Save grid map as CSV
    np.savetxt(f"{output_dir}/{base_name}_grid_{timestamp}.csv", grid_map, delimiter=',', fmt='%d')
    
    # Save route visualization
    fig = visualize_grid_map(grid_map, title=f"Route Grid Map" + 
                               (f" (Difficulty: {predicted_difficulty})" if predicted_difficulty else ""))
    fig.savefig(f"{output_dir}/{base_name}_route_map_{timestamp}.png")
    plt.close(fig)
    
    print(f"Results saved to {output_dir}/ folder")
    return f"{output_dir}/{base_name}_holds_{timestamp}.txt"

def main():
    """
    Main function to run when script is executed directly
    """
    # Create output directory
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 50)
    print("CLIMBING ROUTE ANALYZER")
    print("=" * 50)
    
    # Get image path
    default_img = "climbing_wall.jpg"
    image_path = input(f"Enter path to image (default: {default_img}): ")
    if not image_path:
        image_path = default_img
    
    # Get color
    color_options = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]
    print("\nAvailable colors:", ", ".join(color_options))
    target_color = input("Enter color to detect (default: red): ").lower()
    if not target_color or target_color not in color_options:
        target_color = "red"
    
    # Get sensitivity level
    sensitivity_str = input("Enter sensitivity level (0-100, default: 25): ")
    sensitivity = int(sensitivity_str) if sensitivity_str.isdigit() else 25
    sensitivity = max(0, min(100, sensitivity))  # Clamp between 0-100
    
    # Get minimum area
    min_area_str = input("Enter minimum area to consider as a hold (default: 200): ")
    min_area = int(min_area_str) if min_area_str.isdigit() else 200
    
    # Ask about loading the difficulty prediction model
    use_model = input("Use difficulty prediction model? (y/n, default: y): ").lower() != 'n'
    
    model = None
    if use_model:
        model_path = input("Enter model path (leave blank to download/use demo): ")
        model_path = model_path if model_path else None
        model = load_model(model_path)
    
    try:
        # Run detection and classification
        print("\nProcessing image...")
        holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
            image_path, 
            target_color=target_color,
            sensitivity=sensitivity,
            min_area=min_area
        )
        
        # Predict difficulty if model is available
        predicted_difficulty = None
        if model and len(holds_info) > 0:
            predicted_difficulty = predict_route_difficulty(model, grid_map)
            print(f"\nPredicted route difficulty: {predicted_difficulty}")
        
        # Print results to console
        print(f"\nDetected {len(holds_info)} holds:")
        for i, hold in enumerate(holds_info):
            print(f"Hold {i+1}: {hold['type']} at grid position {hold['grid_position']} "
                 f"(Points: {hold.get('points_value', 1)})")
        
        # Extract cropped region
        x, y, w, h = cropped_region
        original_image = cv2.imread(image_path)
        cropped_image = original_image[y:y+h, x:x+w]
        
        # Display results
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(        original_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 2)
        plt.title("Masked Image")
        plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 3)
        plt.title("Result Image")
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 4)
        plt.title("Cropped Image")
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
        
        # Save results
        save_results(image_path, holds_info, grid_map, result_image, masked_image, cropped_region,
                     predicted_difficulty, output_dir)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
