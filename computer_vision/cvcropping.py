import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def detect_and_classify_holds(image_path, target_color='red', sensitivity=25, min_area=200, max_points_per_hold=1):
    """
    Detect climbing holds of a specified color and classify them
    
    Args:
        image_path (str): Path to the input image
        target_color (str): Color to detect (red, blue, green, yellow, etc.)
        sensitivity (int): Color detection sensitivity (0-100, higher = more lenient)
        min_area (int): Minimum contour area to consider as a hold
        max_points_per_hold (int): Maximum number of grid points to mark per hold (reduces crowding)
        
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
    # Hue: 0-180 (OpenCV uses 0-180 instead of 0-360 for hue)
    # Saturation: 0-255
    # Value: 0-255 (brightness)
    color_ranges = {
        'red': (np.array([0, 120, 70]), np.array([10 + sensitivity//5, 255, 255])),  # Lower red range
        'red2': (np.array([170 - sensitivity//5, 120, 70]), np.array([180, 255, 255])),  # Upper red range
        'blue': (np.array([100 - sensitivity//5, 100, 50]), np.array([130 + sensitivity//5, 255, 255])),
        
        # Improved green range - expanded to detect darker and lighter greens
        'green': (np.array([35 - sensitivity//5, 40, 30]), np.array([90 + sensitivity//5, 255, 255])),
        
        'yellow': (np.array([20 - sensitivity//5, 100, 100]), np.array([35 + sensitivity//5, 255, 255])),
        'orange': (np.array([10 - sensitivity//5, 100, 100]), np.array([25 + sensitivity//5, 255, 255])),
        
        # Improved purple range - adjusted to detect both darker and brighter purples
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for visualization
    result_image = image.copy()
    
    # Filter contours by minimum area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    
    # If no valid contours found
    if not valid_contours:
        print(f"No valid holds detected with current settings.")
        cropped_region = (0, 0, image.shape[1], image.shape[0])  # Full image
        holds_info = []
        grid_map = np.zeros((18, 12), dtype=np.int32)
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
    
    # Create a 12x18 grid map (VERTICAL orientation - 12 columns, 18 rows) for the cropped region
    grid_map = np.zeros((40, 26), dtype=np.int32)
    
    # Calculate cell dimensions based on the cropped region
    cell_width = crop_w / 26  # 12 columns
    cell_height = crop_h / 40  # 18 rows
    
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
        
        # Map center to grid coordinates (with vertical orientation)
        grid_x = max(0, min(25, int(rel_center_x / cell_width)))
        grid_y = max(0, min(39, int(rel_center_y / cell_height)))
        
        # Check if hold is within the cropped region
        if 0 <= rel_x < crop_w and 0 <= rel_y < crop_h:
            # Mark just the center point or a limited number of points
            if max_points_per_hold == 1:
                # Mark just the center point on the grid
                grid_map[grid_y, grid_x] = 1
            else:
                # Mark limited number of points per hold
                points_marked = 0
                for gx in range(max(0, grid_x-1), min(12, grid_x+2)):
                    for gy in range(max(0, grid_y-1), min(18, grid_y+2)):
                        # Only mark if within max_points_per_hold limit and prioritize center
                        if points_marked < max_points_per_hold:
                            dist = (gx - grid_x)**2 + (gy - grid_y)**2  # Distance from center
                            if dist == 0 or points_marked < max_points_per_hold - 1:
                                grid_map[gy, gx] = 1
                                points_marked += 1
            
            # Classify hold - adjusted for smaller holds
            hold_type = classify_hold(cnt, area_adjustment_factor=0.5)
            
            # Store hold information
            holds_info.append({
                "type": hold_type,
                "position": (center_x, center_y),
                "relative_position": (rel_center_x, rel_center_y),
                "grid_position": (grid_x, grid_y),
                "dimensions": (w_cnt, h_cnt),
                "area": area
            })
            
            # Draw bounding box and label on result image
            cv2.rectangle(result_image, (x_cnt, y_cnt), (x_cnt+w_cnt, y_cnt+h_cnt), (0, 255, 0), 2)
            cv2.putText(result_image, hold_type, (x_cnt, y_cnt - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw the grid position on the result image
            cv2.circle(result_image, (center_x, center_y), 3, (0, 255, 255), -1)
            cv2.putText(result_image, f"({grid_x},{grid_y})", (center_x + 5, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw the cropping rectangle on the result image
    cv2.rectangle(result_image, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 2)
    
    return holds_info, grid_map, masked_image, result_image, cropped_region

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
    
    # Extended classification based on multiple features - adjusted for smaller holds
    if area < micro_threshold:
        return "micro"
    elif area > jug_threshold and convexity > 0.85:
        return "jug"
    elif aspect_ratio > 1.5 and solidity > 0.8:
        return "rail"
    elif circularity > 0.75 and convexity > 0.85:  # Relaxed criteria
        return "pinch"
    elif len(approx) > 6 and solidity < 0.85:  # Relaxed criteria
        return "sloper"
    elif aspect_ratio < 1.3 and convexity < 0.8:  # Relaxed criteria
        return "crimp"
    elif 0.7 < convexity < 0.9 and 0.65 < solidity < 0.9:  # Relaxed criteria
        return "pocket"
    else:
        return "foothold"

def debug_color_detection(image_path, target_color, sensitivity=25):
    """
    Debug function to visualize HSV color ranges
    
    Args:
        image_path: Path to the image
        target_color: Color to detect
        sensitivity: Sensitivity level
        
    Returns:
        tuple: Color range information and visualization
    """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    color_ranges = {
        'red': (np.array([0, 120, 70]), np.array([10 + sensitivity//5, 255, 255])),
        'red2': (np.array([170 - sensitivity//5, 120, 70]), np.array([180, 255, 255])),
        'blue': (np.array([100 - sensitivity//5, 100, 50]), np.array([130 + sensitivity//5, 255, 255])),
        'green': (np.array([35 - sensitivity//5, 40, 40]), np.array([90 + sensitivity//5, 255, 255])),
        'yellow': (np.array([20 - sensitivity//5, 100, 100]), np.array([35 + sensitivity//5, 255, 255])),
        'orange': (np.array([10 - sensitivity//5, 100, 100]), np.array([25 + sensitivity//5, 255, 255])),
        'purple': (np.array([125 - sensitivity//5, 40, 40]), np.array([165 + sensitivity//5, 255, 255])),
        'black': (np.array([0, 0, 0]), np.array([180, 70, 40 + sensitivity//2])),
        'white': (np.array([0, 0, 180 - sensitivity*2]), np.array([180, 40 + sensitivity//2, 255]))
    }
    
    # Get the color range
    if target_color == 'red':
        lower_bound1, upper_bound1 = color_ranges['red']
        lower_bound2, upper_bound2 = color_ranges['red2']
        
        mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
        mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        range_info = {
            "red_lower": lower_bound1.tolist(),
            "red_upper": upper_bound1.tolist(),
            "red2_lower": lower_bound2.tolist(),
            "red2_upper": upper_bound2.tolist()
        }
    elif target_color in color_ranges:
        lower_bound, upper_bound = color_ranges[target_color]
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        range_info = {
            f"{target_color}_lower": lower_bound.tolist(),
            f"{target_color}_upper": upper_bound.tolist()
        }
    else:
        raise ValueError(f"Unsupported color: {target_color}")
    
    # Create result image
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return range_info, result

def save_results(image_path, holds_info, grid_map, result_image, masked_image, cropped_region, output_dir="results"):
    """
    Save detection results to files
    
    Args:
        image_path (str): Path to the original image
        holds_info (list): List of detected holds information
        grid_map (numpy.ndarray): The binary grid map
        result_image: Annotated image with hold classifications
        masked_image: Image with only the target color visible
        cropped_region (tuple): (x, y, w, h) of the cropped region
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
        f.write("-" * 50 + "\n")
        for i, hold in enumerate(holds_info):
            f.write(f"Hold {i+1}: {hold['type']}\n")
            f.write(f"  Position in original: {hold['position']}\n")
            f.write(f"  Position in crop: {hold['relative_position']}\n")
            f.write(f"  Grid position: {hold['grid_position']}\n")
            f.write(f"  Size: {hold['dimensions']}\n")
            f.write(f"  Area: {hold['area']} pixels\n")
            f.write("-" * 50 + "\n")
    
    # Save grid map as CSV
    np.savetxt(f"{output_dir}/{base_name}_grid_{timestamp}.csv", grid_map, delimiter=',', fmt='%d')
    
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
    
    # Simple UI for selecting parameters
    print("=" * 50)
    print("CLIMBING HOLDS DETECTOR")
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
    
    # Get points per hold (new parameter)
    points_str = input("Enter max points per hold on grid (1=center only, default: 1): ")
    points_per_hold = int(points_str) if points_str.isdigit() else 1
    points_per_hold = max(1, min(9, points_per_hold))  # Clamp between 1-9
    
    # Debug mode option
    debug_mode = input("Enter 'd' for debug mode (shows color ranges) or any key to continue: ").lower() == 'd'
    
    if debug_mode:
        # Run debug color detection
        print(f"\nDebugging {target_color} color detection with sensitivity {sensitivity}...")
        range_info, debug_image = debug_color_detection(image_path, target_color, sensitivity)
        
        # Print color ranges
        print("\nColor Range Information:")
        for key, value in range_info.items():
            print(f"  {key}: {value}")
        
        # Display debug image
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title(f"{target_color.capitalize()} Color Detection (Sensitivity: {sensitivity})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save debug image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(f"{output_dir}/{base_name}_debug_{target_color}_{timestamp}.jpg", debug_image)
        
        # Ask if user wants to continue with detection
        continue_detection = input("\nContinue with detection? (y/n, default: y): ").lower() != 'n'
        if not continue_detection:
            return 0
    
    try:
        # Run detection and classification
        print("\nProcessing image...")
        holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
            image_path, 
            target_color=target_color,
            sensitivity=sensitivity,
            min_area=min_area,
            max_points_per_hold=points_per_hold
        )
        
        # Print results to console
        print(f"\nDetected {len(holds_info)} holds:")
        for i, hold in enumerate(holds_info):
            print(f"Hold {i+1}: {hold['type']} at grid position {hold['grid_position']}")
        
        # Extract cropped region
        x, y, w, h = cropped_region
        original_image = cv2.imread(image_path)
        cropped_image = original_image[y:y+h, x:x+w]
        
        # Display results
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title(f"Detected {target_color.capitalize()} Holds")
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title(f"{target_color.capitalize()} Color Mask")
        plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title("Cropped Region")
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.title("18x12 Grid Map (Vertical)")
        plt.imshow(grid_map, cmap='binary', interpolation='nearest')
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        # Add grid lines explicitly
        for i in range(grid_map.shape[0] + 1):
            plt.axhline(y=i-0.5, color='gray', linestyle='-', linewidth=0.5)
        for i in range(grid_map.shape[1] + 1):
            plt.axvline(x=i-0.5, color='gray', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save results
        result_file = save_results(image_path, holds_info, grid_map, result_image, masked_image, cropped_region, output_dir)
        print(f"\nDetailed results saved to: {result_file}")
        
        # Show plot
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()
    