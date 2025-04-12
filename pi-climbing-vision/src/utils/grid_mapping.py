from typing import List, Tuple
import numpy as np

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