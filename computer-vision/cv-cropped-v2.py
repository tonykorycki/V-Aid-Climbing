import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.cluster import KMeans  # Added for color clustering

def detect_and_classify_holds(image_path, target_color='red', sensitivity=25, min_area=200, max_points_per_hold=1, 
                             use_adaptive_threshold=True, use_color_clustering=True):
    """
    Detect climbing holds of a specified color and classify them
    
    Args:
        image_path (str): Path to the input image
        target_color (str): Color to detect (red, blue, green, yellow, etc.)
        sensitivity (int): Color detection sensitivity (0-100, higher = more lenient)
        min_area (int): Minimum contour area to consider as a hold
        max_points_per_hold (int): Maximum number of grid points to mark per hold (reduces crowding)
        use_adaptive_threshold (bool): Whether to use adaptive thresholding for better edge detection
        use_color_clustering (bool): Whether to use color clustering for improved color detection
        
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
    
    # Apply bilateral filter to preserve edges while reducing noise
    hsv = cv2.bilateralFilter(hsv, 9, 75, 75)
    
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
    
    # Handle color detection based on method
    if use_color_clustering and target_color != 'black' and target_color != 'white':
        # Use color clustering for more robust detection
        mask = color_clustering_detection(image, hsv, target_color, sensitivity)
    else:
        # Use traditional HSV thresholding
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
    
    # Improve edge detection with adaptive thresholding if enabled
    if use_adaptive_threshold:
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        # Combine with color mask to improve contour detection
        mask = cv2.bitwise_and(mask, thresh)
        
    # Apply mask to get only the target color
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Find contours of holds
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for visualization
    result_image = image.copy()
    
    # Filter contours by minimum area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    
    # If no valid contours found
    if not valid_contours:
        print(f"No valid holds detected with current settings.")
        cropped_region = (0, 0, image.shape[1], image.shape[0])  # Full image
        holds_info = []
        grid_map = np.zeros((40, 26), dtype=np.int32)
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
    
    # Create a 26x40 grid map (VERTICAL orientation - 26 columns, 40 rows) for the cropped region
    grid_map = np.zeros((40, 26), dtype=np.int32)
    
    # Calculate cell dimensions based on the cropped region
    cell_width = crop_w / 26
    cell_height = crop_h / 40
    
    # Process each contour (potential hold)
    holds_info = []
    for i, cnt in enumerate(valid_contours):
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
                for gx in range(max(0, grid_x-1), min(26, grid_x+2)):
                    for gy in range(max(0, grid_y-1), min(40, grid_y+2)):
                        # Only mark if within max_points_per_hold limit and prioritize center
                        if points_marked < max_points_per_hold:
                            dist = (gx - grid_x)**2 + (gy - grid_y)**2  # Distance from center
                            if dist == 0 or points_marked < max_points_per_hold - 1:
                                grid_map[gy, gx] = 1
                                points_marked += 1
            
            # Extract the hold image for texture analysis
            hold_img = extract_hold_image(image, cnt)
            
            # Enhanced hold classification with texture and shape features
            hold_type = enhanced_classify_hold(cnt, area, hold_img)
            
            # Store hold information
            holds_info.append({
                "id": i + 1,
                "type": hold_type,
                "position": (center_x, center_y),
                "relative_position": (rel_center_x, rel_center_y),
                "grid_position": (grid_x, grid_y),
                "dimensions": (w_cnt, h_cnt),
                "area": area
            })
            
            # Draw bounding box and label on result image
            cv2.rectangle(result_image, (x_cnt, y_cnt), (x_cnt+w_cnt, y_cnt+h_cnt), (0, 255, 0), 2)
            cv2.putText(result_image, f"{i+1}: {hold_type}", (x_cnt, y_cnt - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw the grid position on the result image
            cv2.circle(result_image, (center_x, center_y), 3, (0, 255, 255), -1)
            cv2.putText(result_image, f"({grid_x},{grid_y})", (center_x + 5, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw the cropping rectangle on the result image
    cv2.rectangle(result_image, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 0, 255), 2)
    
    # Draw grid lines on the result image for better visualization
    for i in range(1, 26):
        x_pos = crop_x + int(i * cell_width)
        cv2.line(result_image, (x_pos, crop_y), (x_pos, crop_y + crop_h), (100, 100, 100), 1)
    
    for i in range(1, 40):
        y_pos = crop_y + int(i * cell_height)
        cv2.line(result_image, (crop_x, y_pos), (crop_x + crop_w, y_pos), (100, 100, 100), 1)
    
    return holds_info, grid_map, masked_image, result_image, cropped_region

def color_clustering_detection(image, hsv_image, target_color, sensitivity=25):
    """
    Detect color using K-means clustering for more robust detection
    
    Args:
        image: Original BGR image
        hsv_image: HSV converted image
        target_color: Target color to detect
        sensitivity: Detection sensitivity
        
    Returns:
        mask: Binary mask of detected color regions
    """
    # Reshape the image for clustering
    pixel_data = hsv_image.reshape((-1, 3)).astype(np.float32)
    
    # Perform K-means clustering (with k=5 clusters)
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Color target values in HSV for different colors
    color_targets = {
        'red': [(0, 170, 150), (180, 170, 150)],  # Red wraps around hue spectrum
        'blue': [(120, 170, 150)],
        'green': [(60, 150, 120)],
        'yellow': [(30, 200, 200)],
        'orange': [(15, 200, 200)],
        'purple': [(140, 150, 120)]
    }
    
    # Initialize mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Map target color to cluster centers
    centers = centers.astype(np.uint8)
    if target_color in color_targets:
        targets = color_targets[target_color]
        
        # Find clusters closest to target color(s)
        for target_hsv in targets:
            min_dist = float('inf')
            best_cluster = None
            
            # Convert target to HSV array
            target_array = np.array(target_hsv, dtype=np.uint8)
            
            # Find closest cluster
            for i, center in enumerate(centers):
                # Calculate color distance with emphasis on hue
                if target_color == 'red':
                    # Special handling for red (circular hue)
                    h_dist = min(
                        abs(center[0] - target_array[0]),
                        abs((center[0] + 180) % 180 - target_array[0])
                    )
                else:
                    h_dist = abs(center[0] - target_array[0])
                
                s_dist = abs(center[1] - target_array[1])
                v_dist = abs(center[2] - target_array[2])
                
                # Weight hue more heavily
                dist = h_dist * 4 + s_dist + v_dist
                
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = i
            
            # Adjust sensitivity factor based on the input sensitivity
            sensitivity_factor = 1 + (sensitivity / 25)
            
            # Threshold for acceptance as target color (lower = more strict)
            mask_cluster = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_cluster[labels.reshape(image.shape[:2]) == best_cluster] = 255
            mask = cv2.bitwise_or(mask, mask_cluster)
    
            return mask

def extract_hold_image(image, contour):
    """Extract the image region containing a hold for texture analysis"""
    # Create a minimum enclosing rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add a small margin
    margin = 5
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)
    
    # Extract the hold region
    hold_img = image[y1:y2, x1:x2]
    
    return hold_img

def enhanced_classify_hold(contour, area, hold_img):
    """
    Classify holds using enhanced shape, area, and texture features
    
    Args:
        contour: Hold contour
        area: Contour area
        hold_img: Extracted hold image
        
    Returns:
        str: Hold type classification
    """
    # Calculate shape features
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:  # Handle potential division by zero
        return "Unknown"
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Get bounding rectangle dimensions
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area != 0 else 0
    
    # Convex hull analysis
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Texture analysis (gradient-based)
    texture_score = 0
    if hold_img.size > 0:  # Ensure the image is not empty
        gray_hold = cv2.cvtColor(hold_img, cv2.COLOR_BGR2GRAY)
        # Calculate gradient magnitude using Sobel operators
        sobelx = cv2.Sobel(gray_hold, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_hold, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        texture_score = np.mean(gradient_mag) if gradient_mag.size > 0 else 0
    
    # Classification logic based on shape and texture features
    if circularity > 0.8:
        if area < 1000:
            return "Crimp"
        else:
            return "Jug"
    elif aspect_ratio > 2 or aspect_ratio < 0.5:
        return "Sloper"
    elif solidity < 0.8:
        if texture_score > 30:
            return "Pinch"
        else:
            return "Pocket"
    elif extent < 0.7:
        return "Edge"
    else:
        return "Hold"  # Generic hold

def generate_output_filenames(image_path, color):
    """Generate standardized output filenames based on input image and target color"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return {
        "masked": f"{base_name}_{color}_masked_{timestamp}.jpg",
        "result": f"{base_name}_{color}_result_{timestamp}.jpg",
        "data": f"{base_name}_{color}_data_{timestamp}.txt"
    }

def save_results(holds_info, grid_map, masked_image, result_image, output_dir, filenames):
    """Save all output files to the specified directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images
    cv2.imwrite(os.path.join(output_dir, filenames["masked"]), masked_image)
    cv2.imwrite(os.path.join(output_dir, filenames["result"]), result_image)
    
    # Save data file with holds information and grid map
    with open(os.path.join(output_dir, filenames["data"]), 'w') as f:
        f.write(f"Holds Info ({len(holds_info)} holds detected):\n")
        for hold in holds_info:
            f.write(f"ID: {hold['id']}, Type: {hold['type']}, " +
                    f"Grid Position: {hold['grid_position']}, " +
                    f"Dimensions: {hold['dimensions']}, Area: {hold['area']}\n")
        
        f.write("\nGrid Map (26x40):\n")
        for row in grid_map:
            f.write(''.join(str(int(cell)) for cell in row) + '\n')

def display_results(holds_info, masked_image, result_image):
    """Display the results with matplotlib for interactive viewing"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.title("Color Masked Image")
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title(f"Hold Detection Result ({len(holds_info)} holds)")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.show()
    
    # Print hold information summary
    print(f"Detected {len(holds_info)} holds:")
    hold_types = {}
    for hold in holds_info:
        hold_type = hold['type']
        if hold_type in hold_types:
            hold_types[hold_type] += 1
        else:
            hold_types[hold_type] = 1
    
    for hold_type, count in hold_types.items():
        print(f"  {hold_type}: {count}")

def main():
    """Main function to run the hold detection program"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect and classify climbing holds of a specified color.')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--color', default='red', help='Target color to detect (red, blue, green, yellow, orange, purple, black, white)')
    parser.add_argument('--sensitivity', type=int, default=25, help='Color detection sensitivity (0-100, higher = more lenient)')
    parser.add_argument('--min-area', type=int, default=200, help='Minimum contour area to consider as a hold')
    parser.add_argument('--max-points', type=int, default=1, help='Maximum grid points per hold (reduces crowding)')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--no-adaptive', action='store_false', dest='adaptive', help='Disable adaptive thresholding')
    parser.add_argument('--no-clustering', action='store_false', dest='clustering', help='Disable color clustering')
    parser.add_argument('--no-display', action='store_false', dest='display', help='Disable result display')
    
    args = parser.parse_args()
    
    try:
        # Run detection
        holds_info, grid_map, masked_image, result_image, crop_region = detect_and_classify_holds(
            args.image_path, 
            target_color=args.color,
            sensitivity=args.sensitivity,
            min_area=args.min_area,
            max_points_per_hold=args.max_points,
            use_adaptive_threshold=args.adaptive,
            use_color_clustering=args.clustering
        )
        
        # Generate and save results
        filenames = generate_output_filenames(args.image_path, args.color)
        save_results(holds_info, grid_map, masked_image, result_image, args.output_dir, filenames)
        
        # Display results if enabled
        if args.display:
            display_results(holds_info, masked_image, result_image)
            
        print(f"Processing complete. Results saved to {args.output_dir}/")
        print(f"Detected {len(holds_info)} {args.color} holds")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    # Skip the command-line parsing and just run with a specific file
    try:
        # Ask the user for the image file
        image_path = input("Enter the path to your climbing wall image: ") or "climbing_wall.jpg"
        color = input("Enter the color to detect (red, blue, green, yellow, orange, purple, black, white) [default: red]: ") or "red"
        sensitivity = int(input("Enter color sensitivity (0-100) [default: 25]: ") or "25")
        min_area = int(input("Enter minimum hold area [default: 200]: ") or "200")
        output_dir = input("Enter output directory [default: output]: ") or "output"
        
        
            # Run detection
        holds_info, grid_map, masked_image, result_image, crop_region = detect_and_classify_holds(
            image_path, 
            target_color=color,
            sensitivity=sensitivity,
            min_area=min_area,
            max_points_per_hold=1,
            use_adaptive_threshold=True,
            use_color_clustering=True
        )
            
        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save results
        filenames = generate_output_filenames(image_path, color)
        save_results(holds_info, grid_map, masked_image, result_image, output_dir, filenames)
        
        # Display results
        display_results(holds_info, masked_image, result_image)
            
        print(f"Processing complete. Results saved to {output_dir}/")
        print(f"Detected {len(holds_info)} {color} holds")
        
    except Exception as e:
        print(f"Error: {str(e)}")