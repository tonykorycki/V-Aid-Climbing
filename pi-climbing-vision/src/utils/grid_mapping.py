from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_grid_map(holds_info: List[dict], grid_size: Tuple[int, int] = (12, 12)) -> np.ndarray:
    """
    Create a 12x12 grid map based on the detected holds' positions.

    Args:
        holds_info: List of dictionaries containing hold information, including their positions.
        grid_size: Tuple indicating the size of the grid (default is 12x12).

    Returns:
        grid_map: A numpy array representing the grid, where:
            - 0 indicates no hold,
            - 1 indicates a small hold,
            - 2 indicates a large hold or volume.
    """
    grid_map = np.zeros(grid_size, dtype=np.int32)
    
    for hold in holds_info:
        cx, cy = hold["position"]
        grid_x = int(cx * grid_size[0] / 640)  # Assuming image width is 640
        grid_y = int(cy * grid_size[1] / 480)  # Assuming image height is 480
        
        if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
            if hold["points_value"] == 2:
                grid_map[grid_y, grid_x] = 2  # Large hold
            else:
                grid_map[grid_y, grid_x] = 1  # Small hold

    return grid_map

def fill_grid_points_by_shape(grid_map, grid_x, grid_y, points_to_fill, bbox, crop_region, cell_width, cell_height):
    """
    Fill grid points based on hold shape and orientation.
    
    Args:
        grid_map: 12x12 numpy array to fill
        grid_x, grid_y: Grid coordinates
        points_to_fill: 1 for hold, 2 for volume
        bbox: Bounding box coordinates
        crop_region: Crop region coordinates
        cell_width, cell_height: Dimensions of grid cells
    """
    # Place center point
    grid_map[grid_y, grid_x] = points_to_fill
    
    # For volumes, add a second point based on orientation
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
        
        # Make sure the second point is within grid bounds
        second_x = max(0, min(11, second_x))
        second_y = max(0, min(11, second_y))
        grid_map[second_y, second_x] = points_to_fill

def filter_isolated_holds(holds_info: List[dict], threshold: float) -> List[dict]:
    """
    Filter out holds that are isolated based on a distance threshold.

    Args:
        holds_info: List of dictionaries containing hold information.
        threshold: Minimum distance to consider holds as connected.

    Returns:
        filtered_holds: List of holds that are not isolated.
    """
    filtered_holds = []
    for hold in holds_info:
        is_isolated = True
        for other_hold in holds_info:
            if hold != other_hold:
                distance = np.linalg.norm(np.array(hold["position"]) - np.array(other_hold["position"]))
                if distance < threshold:
                    is_isolated = False
                    break
        if not is_isolated:
            filtered_holds.append(hold)
    
    return filtered_holds

def filter_isolated_volumes(holds_info, threshold):
    """
    Remove volumes that don't have any holds nearby.
    
    Args:
        holds_info: List of hold dictionaries
        threshold: Distance threshold
        
    Returns:
        Filtered list of holds
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

def visualize_grid_map(grid_map, title="Route Grid Map", complexity_metrics=None):
    """
    Create a visualization of the route grid map.
    
    Args:
        grid_map: 12x12 numpy array
        title: Plot title
        complexity_metrics: Optional metrics dictionary
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(8, 8))
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
    
    # Add complexity metrics if provided
    if complexity_metrics:
        info_text = f"Density: {complexity_metrics['density']:.2f}, " \
                   f"Holds: {complexity_metrics['num_holds']}, " \
                   f"Volumes: {complexity_metrics['num_volumes']}"
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=9)
    
    plt.tight_layout()
    return fig

def create_route_visualization(image_path, holds_info, grid_map, result_image, masked_image, 
                              cropped_region, predicted_difficulty=None):
    """
    Create a comprehensive visualization of the detected route.
    
    Args:
        image_path: Path to original image
        holds_info: List of hold dictionaries
        grid_map: 12x12 numpy array
        result_image: Annotated image
        masked_image: Image with only detected holds
        cropped_region: Crop region coordinates
        predicted_difficulty: Optional difficulty prediction
        
    Returns:
        Matplotlib figure
    """
    # Read original image
    original_image = cv2.imread(image_path)
    x, y, w, h = cropped_region
    cropped_image = original_image[y:y+h, x:x+w]
    
    # Calculate complexity metrics
    from utils.detection import analyze_route_complexity
    complexity_metrics = analyze_route_complexity(grid_map)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 2)
    plt.title("Detected Holds")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 3)
    plt.title("Masked Holds")
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 4)
    plt.title("Cropped Region")
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 5)
    plt.title("Grid Map")
    plt.imshow(grid_map, cmap='Blues', interpolation='nearest')
    # Draw grid lines and cell values
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    for i in range(grid_map.shape[0] + 1):
        plt.axhline(y=i - 0.5, color='gray', linestyle='-', linewidth=0.5)
    for i in range(grid_map.shape[1] + 1):
        plt.axvline(x=i - 0.5, color='gray', linestyle='-', linewidth=0.5)
    for i in range(grid_map.shape[0]):
        for j in range(grid_map.shape[1]):
            if grid_map[i, j] > 0:
                plt.text(j, i, str(grid_map[i, j]),
                        ha="center", va="center",
                        color="white" if grid_map[i, j] > 1 else "black")
    
    plt.subplot(2, 3, 6)
    plt.title("Route Metrics")
    plt.axis('off')
    
    info_text = ""
    if predicted_difficulty:
        info_text += f"Predicted Difficulty: {predicted_difficulty}\n\n"
    
    info_text += f"Hold Count: {complexity_metrics['num_holds']}\n"
    info_text += f"Volume Count: {complexity_metrics['num_volumes']}\n"
    info_text += f"Density: {complexity_metrics['density']:.2f}\n"
    info_text += f"Avg Distance: {complexity_metrics['avg_distance']:.2f}\n"
    info_text += f"Max Vertical Gap: {complexity_metrics['max_vertical_gap']}"
    
    plt.text(0.5, 0.5, info_text, 
             ha='center', va='center', 
             fontsize=11, 
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    return fig