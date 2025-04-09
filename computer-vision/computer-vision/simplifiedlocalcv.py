#just run from IDE with prompts

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def detect_and_classify_holds(image_path, target_color='red', sensitivity=25, min_area=200):
    """
    Detect climbing holds of a specified color and classify them
    
    Args:
        image_path (str): Path to the input image
        target_color (str): Color to detect (red, blue, green, yellow, etc.)
        sensitivity (int): Color detection sensitivity
        min_area (int): Minimum contour area to consider as a hold
        
    Returns:
        tuple: (holds_info, grid_map, result_image)
    """
    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges based on common climbing hold colors
    color_ranges = {
        'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),  # Lower red range
        'red2': (np.array([170, 100, 100]), np.array([180, 255, 255])),  # Upper red range
        'blue': (np.array([100, 100, 100]), np.array([130, 255, 255])),
        'green': (np.array([40, 100, 100]), np.array([80, 255, 255])),
        'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
        'orange': (np.array([10, 100, 100]), np.array([25, 255, 255])),
        'purple': (np.array([130, 50, 100]), np.array([160, 255, 255])),
        'black': (np.array([0, 0, 0]), np.array([180, 255, 50])),
        'white': (np.array([0, 0, 200]), np.array([180, 30, 255]))
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
    
    # Create a 12x18 grid map (VERTICAL orientation - 12 columns, 18 rows)
    height, width, _ = image.shape
    grid_map = np.zeros((18, 12), dtype=np.int32)  # Note the reversed dimensions for vertical
    
    # Calculate cell dimensions
    cell_width = width / 12  # 12 columns
    cell_height = height / 18  # 18 rows
    
    # Process each contour (potential hold)
    holds_info = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # Skip small contours (noise)
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Map to grid coordinates (now with vertical orientation)
        grid_x_start = max(0, min(11, int(x / cell_width)))  # Column (horizontal position)
        grid_x_end = max(0, min(11, int((x + w) / cell_width)))
        grid_y_start = max(0, min(17, int(y / cell_height)))  # Row (vertical position)
        grid_y_end = max(0, min(17, int((y + h) / cell_height)))
        
        # Mark cells in grid
        for gx in range(grid_x_start, grid_x_end + 1):
            for gy in range(grid_y_start, grid_y_end + 1):
                grid_map[gy, gx] = 1
        
        # Classify hold
        hold_type = classify_hold(cnt)
        
        # Calculate center point
        center_x, center_y = x + w//2, y + h//2
        
        # Store hold information
        holds_info.append({
            "type": hold_type,
            "position": (center_x, center_y),
            "grid_position": (grid_x_start, grid_y_start),
            "dimensions": (w, h),
            "area": area
        })
        
        # Draw bounding box and label on result image
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, hold_type, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return holds_info, grid_map, masked_image, result_image

def classify_hold(cnt):
    """
    Classify hold based on shape features
    
    Args:
        cnt: Contour of the potential hold
        
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
    
    # Extended classification based on multiple features
    if area < 500:
        return "micro"
    elif area > 5000 and convexity > 0.85:
        return "jug"
    elif aspect_ratio > 1.5 and solidity > 0.8:
        return "rail"
    elif circularity > 0.8 and convexity > 0.9:
        return "pinch"
    elif len(approx) > 8 and solidity < 0.8:
        return "sloper"
    elif aspect_ratio < 1.2 and convexity < 0.75:
        return "crimp"
    elif 0.75 < convexity < 0.9 and 0.7 < solidity < 0.9:
        return "pocket"
    else:
        return "foothold"

def save_results(image_path, holds_info, grid_map, result_image, masked_image, output_dir="results"):
    """
    Save detection results to files
    
    Args:
        image_path (str): Path to the original image
        holds_info (list): List of detected holds information
        grid_map (numpy.ndarray): The binary grid map
        result_image: Annotated image with hold classifications
        masked_image: Image with only the target color visible
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
    
    # Save holds information as text
    with open(f"{output_dir}/{base_name}_holds_{timestamp}.txt", "w") as f:
        f.write(f"Detected {len(holds_info)} holds:\n")
        f.write("-" * 50 + "\n")
        for i, hold in enumerate(holds_info):
            f.write(f"Hold {i+1}: {hold['type']}\n")
            f.write(f"  Position: {hold['position']}\n")
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
    
    # Get minimum area
    min_area_str = input("Enter minimum area to consider as a hold (default: 200): ")
    min_area = int(min_area_str) if min_area_str.isdigit() else 200
    
    try:
        # Run detection and classification
        print("\nProcessing image...")
        holds_info, grid_map, masked_image, result_image = detect_and_classify_holds(
            image_path, 
            target_color=target_color,
            min_area=min_area
        )
        
        # Print results to console
        print(f"\nDetected {len(holds_info)} holds:")
        for i, hold in enumerate(holds_info):
            print(f"Hold {i+1}: {hold['type']} at grid position {hold['grid_position']}")
        
        # Display results
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.title(f"Detected {target_color.capitalize()} Holds")
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.title(f"{target_color.capitalize()} Color Mask")
        plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
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
        result_file = save_results(image_path, holds_info, grid_map, result_image, masked_image, output_dir)
        print(f"\nDetailed results saved to: {result_file}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()