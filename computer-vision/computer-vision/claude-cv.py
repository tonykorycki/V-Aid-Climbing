import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
from picamera2 import Picamera2
import threading
import queue

class ClimbingHoldDetector:
    def __init__(self):
        """Initialize the climbing hold detector with default parameters"""
        # Default parameters
        self.sensitivity = 25
        self.min_area = 200
        self.max_points_per_hold = 1
        self.output_dir = "results"
        self.num_frames_to_average = 5  # Default number of frames to average
        self.capture_delay = 0.5  # Delay between captures in seconds
        
        # Color ranges
        self.color_ranges = {}
        self._initialize_color_ranges()
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize Pi Camera
        self.picam = None
        
    def _initialize_color_ranges(self):
        """Initialize color range dictionary with default values"""
        # Base color ranges - will be adjusted by sensitivity later
        self.color_ranges = {
            'red': (np.array([0, 120, 70]), np.array([10, 255, 255])),  # Lower red range
            'red2': (np.array([170, 120, 70]), np.array([180, 255, 255])),  # Upper red range
            'blue': (np.array([100, 100, 50]), np.array([130, 255, 255])),
            'green': (np.array([35, 40, 30]), np.array([90, 255, 255])),
            'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
            'orange': (np.array([10, 100, 100]), np.array([25, 255, 255])),
            'purple': (np.array([125, 40, 40]), np.array([165, 255, 255])),
            'black': (np.array([0, 0, 0]), np.array([180, 70, 40])),
            'white': (np.array([0, 0, 180]), np.array([180, 40, 255]))
        }
    
    def _adjust_color_ranges(self):
        """Adjust color ranges based on current sensitivity"""
        # Copy base ranges
        adjusted_ranges = {}
        
        for color, (lower, upper) in self.color_ranges.items():
            if color == 'red':
                # Lower red range
                new_lower = np.array([
                    max(0, lower[0]),
                    max(0, lower[1]),
                    max(0, lower[2])
                ])
                new_upper = np.array([
                    min(180, upper[0] + self.sensitivity//5),
                    min(255, upper[1]),
                    min(255, upper[2])
                ])
                adjusted_ranges[color] = (new_lower, new_upper)
            elif color == 'red2':
                # Upper red range
                new_lower = np.array([
                    max(0, lower[0] - self.sensitivity//5),
                    max(0, lower[1]),
                    max(0, lower[2])
                ])
                new_upper = np.array([
                    min(180, upper[0]),
                    min(255, upper[1]),
                    min(255, upper[2])
                ])
                adjusted_ranges[color] = (new_lower, new_upper)
            elif color == 'black':
                new_lower = np.array([
                    lower[0],
                    lower[1],
                    lower[2]
                ])
                new_upper = np.array([
                    upper[0],
                    min(255, upper[1] + self.sensitivity//2),
                    min(255, upper[2] + self.sensitivity//2)
                ])
                adjusted_ranges[color] = (new_lower, new_upper)
            elif color == 'white':
                new_lower = np.array([
                    lower[0],
                    lower[1],
                    max(0, lower[2] - self.sensitivity*2)
                ])
                new_upper = np.array([
                    upper[0],
                    min(255, upper[1] + self.sensitivity//2),
                    upper[2]
                ])
                adjusted_ranges[color] = (new_lower, new_upper)
            else:
                # Standard adjustment for other colors
                new_lower = np.array([
                    max(0, lower[0] - self.sensitivity//5),
                    max(0, lower[1] - self.sensitivity//3),
                    max(0, lower[2] - self.sensitivity//3)
                ])
                new_upper = np.array([
                    min(180, upper[0] + self.sensitivity//5),
                    min(255, upper[1]),
                    min(255, upper[2])
                ])
                adjusted_ranges[color] = (new_lower, new_upper)
                
        return adjusted_ranges
    
    def initialize_camera(self, resolution=(1920, 1080)):
        """Initialize the Pi Camera with specified resolution"""
        self.picam = Picamera2()
        config = self.picam.create_still_configuration(main={"size": resolution})
        self.picam.configure(config)
        self.picam.start()
        time.sleep(2)  # Allow camera to initialize
        print("Camera initialized")
        
    def release_camera(self):
        """Release camera resources"""
        if self.picam:
            self.picam.close()
            print("Camera released")
    
    def capture_and_average(self):
        """Capture multiple frames and average them to reduce noise"""
        print(f"Capturing {self.num_frames_to_average} frames for averaging...")
        frames = []
        
        for i in range(self.num_frames_to_average):
            print(f"Capturing frame {i+1}/{self.num_frames_to_average}")
            frame = self.picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
            frames.append(frame)
            time.sleep(self.capture_delay)
        
        # Average the frames
        avg_frame = np.zeros_like(frames[0], dtype=np.float32)
        for frame in frames:
            avg_frame += frame.astype(np.float32)
        avg_frame /= len(frames)
        avg_frame = avg_frame.astype(np.uint8)
        
        print("Frame averaging complete")
        return avg_frame
    
    def capture_parallel(self, frame_queue, num_frames):
        """Capture frames in parallel to speed up the process"""
        for _ in range(num_frames):
            frame = self.picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_queue.put(frame)
            time.sleep(self.capture_delay)
    
    def capture_and_average_threaded(self):
        """Capture multiple frames using threading for better performance"""
        print(f"Capturing {self.num_frames_to_average} frames with threading...")
        
        frame_queue = queue.Queue()
        capture_thread = threading.Thread(
            target=self.capture_parallel, 
            args=(frame_queue, self.num_frames_to_average)
        )
        
        capture_thread.start()
        frames = []
        
        # Show progress while waiting for captures
        for i in range(self.num_frames_to_average):
            print(f"Waiting for frame {i+1}/{self.num_frames_to_average}")
            frame = frame_queue.get()
            frames.append(frame)
            print(f"Received frame {i+1}/{self.num_frames_to_average}")
        
        capture_thread.join()
        
        # Average the frames
        avg_frame = np.zeros_like(frames[0], dtype=np.float32)
        for frame in frames:
            avg_frame += frame.astype(np.float32)
        avg_frame /= len(frames)
        avg_frame = avg_frame.astype(np.uint8)
        
        print("Frame averaging complete")
        return avg_frame
    
    def detect_and_classify_holds(self, image, target_color='red'):
        """
        Detect climbing holds of a specified color and classify them
        
        Args:
            image: Input image (numpy array)
            target_color (str): Color to detect (red, blue, green, yellow, etc.)
            
        Returns:
            tuple: (holds_info, grid_map, masked_image, result_image, cropped_region)
        """
        # Apply bilateral filter to reduce noise while preserving edges
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get adjusted color ranges
        color_ranges = self._adjust_color_ranges()
        
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
            # Try YCrCb color space for difficult colors
            if target_color in ['white', 'black', 'gray']:
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                
                if target_color == 'white':
                    y_lower = 200 - self.sensitivity
                    mask = cv2.inRange(ycrcb, (y_lower, 0, 0), (255, 255, 255))
                elif target_color == 'black':
                    y_upper = 50 + self.sensitivity//2
                    mask = cv2.inRange(ycrcb, (0, 0, 0), (y_upper, 255, 255))
                else:  # gray
                    y_lower = 70 - self.sensitivity//2
                    y_upper = 180 + self.sensitivity//2
                    mask = cv2.inRange(ycrcb, (y_lower, 120, 120), (y_upper, 140, 140))
            else:
                raise ValueError(f"Unsupported color: {target_color}. Available colors: {list(color_ranges.keys())}")
        
        # Apply adaptive morphological operations based on image size
        kernel_size = max(3, min(7, image.shape[1] // 300))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening (erosion followed by dilation) to remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion) to close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to get only the target color
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Find contours of holds
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy for visualization
        result_image = image.copy()
        
        # Filter contours by minimum area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_area]
        
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
        
        # Create a 26x40 grid map (VERTICAL orientation) for the cropped region
        grid_map = np.zeros((40, 26), dtype=np.int32)
        
        # Calculate cell dimensions based on the cropped region
        cell_width = crop_w / 26
        cell_height = crop_h / 40
        
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
                if self.max_points_per_hold == 1:
                    # Mark just the center point on the grid
                    grid_map[grid_y, grid_x] = 1
                else:
                    # Mark limited number of points per hold
                    points_marked = 0
                    for gx in range(max(0, grid_x-1), min(26, grid_x+2)):
                        for gy in range(max(0, grid_y-1), min(40, grid_y+2)):
                            # Only mark if within max_points_per_hold limit and prioritize center
                            if points_marked < self.max_points_per_hold:
                                dist = (gx - grid_x)**2 + (gy - grid_y)**2  # Distance from center
                                if dist == 0 or points_marked < self.max_points_per_hold - 1:
                                    grid_map[gy, gx] = 1
                                    points_marked += 1
                
                # Classify hold
                hold_type = self.classify_hold(cnt)
                
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
    
    def classify_hold(self, cnt):
        """
        Classify hold based on shape features with adjustment for smaller holds
        
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
        
        # Additional feature: Hu Moments for shape recognition
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments) if moments['m00'] != 0 else np.zeros(7)
        
        # Adjusted area thresholds for smaller holds
        micro_threshold = 500 * 0.5  # area_adjustment_factor = 0.5
        jug_threshold = 5000 * 0.5
        
        # Extended classification based on multiple features and Hu moments
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
        # Use Hu moments for additional classification
        elif abs(hu_moments[0][0]) < 0.2 and circularity > 0.6:
            return "jug" 
        elif abs(hu_moments[0][0]) > 0.3 and aspect_ratio > 1.2:
            return "rail"
        else:
            return "foothold"
    
    def calibrate_color(self, reference_image, num_samples=5):
        """
        Use a reference image to calibrate color detection
        
        Args:
            reference_image: Image containing color references
            num_samples: Number of color samples to take
            
        Returns:
            dict: Calibrated color ranges
        """
        print("Starting color calibration mode...")
        print("Please click on 5 examples of the target color")
        
        # Display the image and get user clicks
        samples = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                samples.append((x, y))
                # Draw a circle around selected point
                cv2.circle(reference_image, (x, y), 5, (0, 255, 0), 2)
                cv2.imshow('Select Color Samples', reference_image)
                print(f"Sample {len(samples)}/{num_samples} collected at ({x}, {y})")
        
        cv2.imshow('Select Color Samples', reference_image)
        cv2.setMouseCallback('Select Color Samples', mouse_callback)
        
        while len(samples) < num_samples:
            cv2.waitKey(100)
        
        cv2.destroyAllWindows()
        
        # Convert image to HSV
        hsv_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV)
        
        # Get HSV values from samples
        hsv_samples = [hsv_image[y, x] for x, y in samples]
        
        # Calculate average and standard deviation
        hsv_samples = np.array(hsv_samples)
        hsv_mean = np.mean(hsv_samples, axis=0)
        hsv_std = np.std(hsv_samples, axis=0)
        
        # Create new range (mean ± 2*std)
        hsv_lower = np.array([
            max(0, hsv_mean[0] - 2 * hsv_std[0]),
            max(0, hsv_mean[1] - 2 * hsv_std[1]),
            max(0, hsv_mean[2] - 2 * hsv_std[2])
        ]).astype(np.uint8)
        
        hsv_upper = np.array([
            min(180, hsv_mean[0] + 2 * hsv_std[0]),
            min(255, hsv_mean[1] + 2 * hsv_std[1]),
            min(255, hsv_mean[2] + 2 * hsv_std[2])
        ]).astype(np.uint8)
        
        # Handle hue wrap-around for reddish colors
        if hsv_mean[0] < 10 or hsv_mean[0] > 170:
            # Likely a red color that wraps around the hue spectrum
            if hsv_mean[0] < 10:
                # Lower red range
                red_lower = np.array([0, hsv_lower[1], hsv_lower[2]])
                red_upper = np.array([hsv_upper[0], hsv_upper[1], hsv_upper[2]])
                # Upper red range
                red2_lower = np.array([170, hsv_lower[1], hsv_lower[2]])
                red2_upper = np.array([180, hsv_upper[1], hsv_upper[2]])
            else:
                # Upper red range
                red2_lower = np.array([hsv_lower[0], hsv_lower[1], hsv_lower[2]])
                red2_upper = np.array([180, hsv_upper[1], hsv_upper[2]])
                # Lower red range
                red_lower = np.array([0, hsv_lower[1], hsv_lower[2]])
                red_upper = np.array([10, hsv_upper[1], hsv_upper[2]])
                
            print(f"Calibrated RED color ranges:")
            print(f"  Lower range: {red_lower} to {red_upper}")
            print(f"  Upper range: {red2_lower} to {red2_upper}")
            
            return {
                'red': (red_lower, red_upper),
                'red2': (red2_lower, red2_upper)
            }
        else:
            print(f"Calibrated color range: {hsv_lower} to {hsv_upper}")
            return {'custom': (hsv_lower, hsv_upper)}
    
    def detect_with_calibrated_colors(self, image, calibrated_ranges):
        """
        Use calibrated color ranges to detect climbing holds
        
        Args:
            image: Input image
            calibrated_ranges: Dictionary of calibrated color ranges
            
        Returns:
            tuple: Detection results
        """
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask from calibrated ranges
        if 'red' in calibrated_ranges and 'red2' in calibrated_ranges:
            # Red special case
            lower1, upper1 = calibrated_ranges['red']
            lower2, upper2 = calibrated_ranges['red2']
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Other colors
            lower, upper = next(iter(calibrated_ranges.values()))
            mask = cv2.inRange(hsv, lower, upper)
        
        # Rest of the detection process as before
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Continue with the standard detection pipeline...
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Find contours of holds
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # The rest of the processing would follow the same pattern as in detect_and_classify_holds
        # ...
        
        return masked_image, contours
    
    def save_results(self, image_path, holds_info, grid_map, result_image, masked_image, cropped_region):
        """
        Save detection results to files
        
        Args:
            image_path (str): Path or identifier for the original image
            holds_info (list): List of detected holds information
            grid_map (numpy.ndarray): The binary grid map
            result_image: Annotated image with hold classifications
            masked_image: Image with only the target color visible
            cropped_region (tuple): (x, y, w, h) of the cropped region
            
        Returns:
            str: Path to the saved holds information file
        """
        # Get base filename without extension or use timestamp if no path
        if isinstance(image_path, str) and os.path.exists(image_path):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            base_name = "picam_capture"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save annotated image
        result_path = f"{self.output_dir}/{base_name}_detected_{timestamp}.jpg"
        cv2.imwrite(result_path, result_image)
        
        # Save masked image
        cv2.imwrite(f"{self.output_dir}/{base_name}_masked_{timestamp}.jpg", masked_image)
        
        # Extract and save the cropped region
        x, y, w, h = cropped_region
        cropped_image = result_image[y:y+h, x:x+w]
        cv2.imwrite(f"{self.output_dir}/{base_name}_cropped_{timestamp}.jpg", cropped_image)
        
        # Save holds information as text
        info_path = f"{self.output_dir}/{base_name}_holds_{timestamp}.txt"
        with open(info_path, "w") as f:
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
        np.savetxt(f"{self.output_dir}/{base_name}_grid_{timestamp}.csv", grid_map, delimiter=',', fmt='%d')
        
        print(f"Results saved to {self.output_dir}/ folder")
        return info_path

    def run_interactive(self):
        """Run the detector in interactive mode with Pi Camera"""
        print("=" * 50)
        print("PI CAMERA CLIMBING HOLDS DETECTOR")
        print("=" * 50)
        
        try:
            # Initialize camera
            self.initialize_camera()
            while True:
                print("\nMenu:")
                print("1. Adjust sensitivity (current: {})".format(self.sensitivity))
                print("2. Adjust minimum area (current: {})".format(self.min_area))
                print("3. Adjust frames to average (current: {})".format(self.num_frames_to_average))
                print("4. Detect holds (default: red)")
                print("5. Detect holds (custom color)")
                print("6. Color calibration mode")
                print("7. Advanced color tuning")
                print("8. Auto white balance adjustment")
                print("9. Quit")
                
                choice = input("Select an option: ")
                
                if choice == '1':
                    try:
                        self.sensitivity = int(input("Enter new sensitivity (5-50): "))
                        self.sensitivity = max(5, min(50, self.sensitivity))
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                elif choice == '2':
                    try:
                        self.min_area = int(input("Enter new minimum area (50-500): "))
                        self.min_area = max(50, min(500, self.min_area))
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                elif choice == '3':
                    try:
                        self.num_frames_to_average = int(input("Enter frames to average (1-10): "))
                        self.num_frames_to_average = max(1, min(10, self.num_frames_to_average))
                    except ValueError:
                        print("Invalid input. Keeping current value.")
                
                elif choice == '4':
                    print("Capturing and processing frames...")
                    frame = self.capture_and_average_threaded()
                    
                    holds_info, grid_map, masked_image, result_image, cropped_region = self.detect_and_classify_holds(frame, 'red')
                    
                    if len(holds_info) > 0:
                        print(f"Detected {len(holds_info)} holds")
                        self.save_results("capture", holds_info, grid_map, result_image, masked_image, cropped_region)
                        
                        # Display results
                        cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
                        cv2.imshow('Results', cv2.resize(result_image, (800, 600)))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                        print("No holds detected. Try adjusting sensitivity or minimum area.")
                        
                elif choice == '5':
                    colors = list(self.color_ranges.keys())
                    colors = [c for c in colors if c != 'red2']  # Remove red2 from the list
                    
                    print("Available colors:")
                    for i, color in enumerate(colors):
                        print(f"{i+1}. {color}")
                        
                    try:
                        color_idx = int(input("Select color (number): ")) - 1
                        if 0 <= color_idx < len(colors):
                            target_color = colors[color_idx]
                            
                            print(f"Detecting {target_color} holds...")
                            frame = self.capture_and_average_threaded()
                            
                            holds_info, grid_map, masked_image, result_image, cropped_region = self.detect_and_classify_holds(frame, target_color)
                            
                            if len(holds_info) > 0:
                                print(f"Detected {len(holds_info)} {target_color} holds")
                                self.save_results(f"capture_{target_color}", holds_info, grid_map, result_image, masked_image, cropped_region)
                                
                                # Display results
                                cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
                                cv2.imshow('Results', cv2.resize(result_image, (800, 600)))
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                            else:
                                print(f"No {target_color} holds detected. Try adjusting sensitivity or minimum area.")
                        else:
                            print("Invalid selection.")
                    except ValueError:
                        print("Invalid input.")
                
                elif choice == '6':
                    print("Entering color calibration mode...")
                    print("Capturing reference frame...")
                    
                    reference_frame = self.capture_and_average_threaded()
                    
                    # Perform color calibration
                    calibrated_ranges = self.calibrate_color(reference_frame)
                    
                    # Test calibration
                    print("Testing calibration...")
                    masked_image, contours = self.detect_with_calibrated_colors(reference_frame, calibrated_ranges)
                    
                    # Show results
                    result_frame = reference_frame.copy()
                    cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)
                    
                    cv2.namedWindow('Calibration Results', cv2.WINDOW_NORMAL)
                    cv2.imshow('Calibration Results', cv2.resize(result_frame, (800, 600)))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    # Ask to save calibration
                    save_cal = input("Save this calibration to a custom color? (y/n): ").lower()
                    if save_cal == 'y':
                        color_name = input("Enter a name for this color: ")
                        if 'custom' in calibrated_ranges:
                            self.color_ranges[color_name] = calibrated_ranges['custom']
                        elif 'red' in calibrated_ranges:
                            self.color_ranges[color_name] = calibrated_ranges['red']
                            self.color_ranges[color_name + '2'] = calibrated_ranges['red2']
                        print(f"Color '{color_name}' added to available colors.")
                
                elif choice == '7':
                    self._advanced_color_tuning()
                
                elif choice == '8':
                    self._auto_white_balance_adjustment()
                
                elif choice == '9':
                    print("Exiting...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
        
        finally:
            # Release camera resources
            self.release_camera()
    
    def _advanced_color_tuning(self):
        """Advanced color tuning interface"""
        print("\nAdvanced Color Tuning")
        print("-------------------")
        
        # List available colors
        colors = list(self.color_ranges.keys())
        for i, color in enumerate(colors):
            print(f"{i+1}. {color}")
        
        try:
            color_idx = int(input("Select color to tune (number): ")) - 1
            if 0 <= color_idx < len(colors):
                color_name = colors[color_idx]
                lower, upper = self.color_ranges[color_name]
                
                print(f"\nCurrent {color_name} range:")
                print(f"Lower bound: H={lower[0]}, S={lower[1]}, V={lower[2]}")
                print(f"Upper bound: H={upper[0]}, S={upper[1]}, V={upper[2]}")
                
                # Interactive tuning
                print("\nCapturing reference image for tuning...")
                frame = self.capture_and_average_threaded()
                
                # Create tuning interface with trackbars
                cv2.namedWindow('Color Tuning', cv2.WINDOW_NORMAL)
                
                # Create trackbars
                cv2.createTrackbar('H Min', 'Color Tuning', lower[0], 180, lambda x: None)
                cv2.createTrackbar('S Min', 'Color Tuning', lower[1], 255, lambda x: None)
                cv2.createTrackbar('V Min', 'Color Tuning', lower[2], 255, lambda x: None)
                cv2.createTrackbar('H Max', 'Color Tuning', upper[0], 180, lambda x: None)
                cv2.createTrackbar('S Max', 'Color Tuning', upper[1], 255, lambda x: None)
                cv2.createTrackbar('V Max', 'Color Tuning', upper[2], 255, lambda x: None)
                
                while True:
                    # Get current trackbar positions
                    h_min = cv2.getTrackbarPos('H Min', 'Color Tuning')
                    s_min = cv2.getTrackbarPos('S Min', 'Color Tuning')
                    v_min = cv2.getTrackbarPos('V Min', 'Color Tuning')
                    h_max = cv2.getTrackbarPos('H Max', 'Color Tuning')
                    s_max = cv2.getTrackbarPos('S Max', 'Color Tuning')
                    v_max = cv2.getTrackbarPos('V Max', 'Color Tuning')
                    
                    # Update color range
                    lower_bound = np.array([h_min, s_min, v_min])
                    upper_bound = np.array([h_max, s_max, v_max])
                    
                    # Apply mask
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower_bound, upper_bound)
                    
                    # Apply morphological operations
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Show masked image
                    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
                    
                    # Display results
                    cv2.imshow('Color Tuning', cv2.resize(masked_image, (800, 600)))
                    
                    # Exit on ESC
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:  # ESC key
                        break
                
                cv2.destroyAllWindows()
                
                # Ask to save changes
                save_changes = input("Save these changes? (y/n): ").lower()
                if save_changes == 'y':
                    self.color_ranges[color_name] = (lower_bound, upper_bound)
                    print(f"Color range for {color_name} updated.")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
    
    def _auto_white_balance_adjustment(self):
        """Perform auto white balance adjustment"""
        print("Performing auto white balance adjustment...")
        
        # Capture frame
        frame = self.capture_and_average_threaded()
        
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Separate L, A, B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        lab = cv2.merge((l, a, b))
        
        # Convert back to BGR
        balanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Display before and after
        comparison = np.hstack((frame, balanced_frame))
        cv2.namedWindow('White Balance Adjustment', cv2.WINDOW_NORMAL)
        cv2.imshow('White Balance Adjustment', cv2.resize(comparison, (1200, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Ask to use this frame for detection
        use_frame = input("Use this balanced image for detection? (y/n): ").lower()
        if use_frame == 'y':
            # Detect holds with the balanced frame
            target_color = input("Enter color to detect (default: red): ") or 'red'
            
            holds_info, grid_map, masked_image, result_image, cropped_region = self.detect_and_classify_holds(balanced_frame, target_color)
            
            if len(holds_info) > 0:
                print(f"Detected {len(holds_info)} {target_color} holds")
                self.save_results(f"balanced_{target_color}", holds_info, grid_map, result_image, masked_image, cropped_region)
                
                # Display results
                cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
                cv2.imshow('Results', cv2.resize(result_image, (800, 600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"No {target_color} holds detected. Try adjusting sensitivity or minimum area.")

    def add_adaptive_color_filtering(self, image, color_name):
        """
        Apply adaptive color filtering based on lighting conditions
        
        Args:
            image: Input image
            color_name: Target color
            
        Returns:
            tuple: (filtered_image, mask)
        """
        # Analyze image brightness and contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        print(f"Image properties - Brightness: {brightness:.2f}, Contrast: {contrast:.2f}")
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get base color range
        if color_name == 'red':
            lower1, upper1 = self.color_ranges['red']
            lower2, upper2 = self.color_ranges['red2']
        else:
            lower, upper = self.color_ranges.get(color_name, (np.array([0, 0, 0]), np.array([180, 255, 255])))
        
        # Adapt saturation and value thresholds based on lighting conditions
        # For low brightness (dark) conditions
        if brightness < 80:
            # Reduce the lower value threshold to detect colors in darker areas
            adjustment = np.array([0, 0, -30])
        # For high brightness (bright) conditions
        elif brightness > 180:
            # Increase saturation threshold to avoid washout
            adjustment = np.array([0, 30, 0])
        # For low contrast conditions
        elif contrast < 30:
            # Widen the hue range slightly
            adjustment = np.array([5, -10, -10])
        else:
            # Normal conditions
            adjustment = np.array([0, 0, 0])
        
        # Apply adjustments
        if color_name == 'red':
            # Apply to both red ranges
            adjusted_lower1 = np.maximum(0, lower1 - adjustment)
            adjusted_upper1 = np.minimum([180, 255, 255], upper1 + adjustment)
            adjusted_lower2 = np.maximum(0, lower2 - adjustment)
            adjusted_upper2 = np.minimum([180, 255, 255], upper2 + adjustment)
            
            # Create masks
            mask1 = cv2.inRange(hsv, adjusted_lower1, adjusted_upper1)
            mask2 = cv2.inRange(hsv, adjusted_lower2, adjusted_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Apply to normal color range
            adjusted_lower = np.maximum(0, lower - adjustment)
            adjusted_upper = np.minimum([180, 255, 255], upper + adjustment)
            
            # Create mask
            mask = cv2.inRange(hsv, adjusted_lower, adjusted_upper)
        
        # Apply additional processing based on lighting conditions
        if brightness < 80:
            # For darker conditions, apply gamma correction
            gamma = 1.5
            lookUpTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
            image = cv2.LUT(image, lookUpTable)
        elif brightness > 180:
            # For brighter conditions, reduce image intensity
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_img[:,:,2] = np.clip(hsv_img[:,:,2] * 0.8, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        
        # Apply the mask to get filtered image
        filtered_image = cv2.bitwise_and(image, image, mask=mask)
        
        return filtered_image, mask

    def main(self, color_to_detect='red'):
        """
        Main function to run the climbing hold detector
        
        Args:
            color_to_detect: Color to detect (default: red)
        """
        detector = ClimbingHoldDetector()
        detector.run_interactive()

if __name__ == "__main__":
    ClimbingHoldDetector().main()