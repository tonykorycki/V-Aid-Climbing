import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def apply_clahe_on_value(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE on the Value channel to improve brightness/contrast.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v = clahe.apply(v)
    hsv_clahe = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

def classify_hold(cnt, area_adjustment_factor=1.0):
    """
    Same advanced shape-based hold classification from your script.
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h) if h != 0 else 0
    
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
    
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0
    
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    solidity = float(area) / hull_area if hull_area != 0 else 0
    
    micro_threshold = 500 * area_adjustment_factor
    jug_threshold = 5000 * area_adjustment_factor
    
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

def detect_and_classify_holds(
    image_path,
    target_color='red',
    min_area=200,
    clip_limit=2.0
):
    """
    Detect climbing holds using multi-range color thresholds, plus
    shape-based classification, then build a (40×26) shape-based grid.
    
    Returns: (holds_info, grid_map, masked_image, result_image, cropped_region)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Enhance brightness & contrast
    enhanced_bgr = apply_clahe_on_value(original, clip_limit=clip_limit)
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)

    # Multiple sub-ranges for each color, especially purple
    strict_color_ranges = {
        'red': [
            # Red can wrap around hue=0 or 180
            (np.array([0, 140, 70]),  np.array([10, 255, 255])),
            (np.array([170, 140, 70]), np.array([180,255,255]))
        ],
        'blue': [
            # You can loosen min S or V if missing dark blues
            (np.array([95, 100, 70]),  np.array([125, 255, 255]))
        ],
        'green': [
            (np.array([35, 90, 80]),   np.array([80, 255, 255]))
        ],
        'yellow': [
            (np.array([20, 140, 100]), np.array([35, 255, 255]))
        ],
        'orange': [
            (np.array([10, 150, 80]),  np.array([25, 255, 255]))
        ],
        'purple': [
            # Extended for large/dark purples. 
            # 1) Lighter purple
            (np.array([125, 40,  50]), np.array([140, 255, 255])),
            # 2) Deeper/darker purple
            (np.array([130, 30, 30]), np.array([170, 255, 255]))
        ],
        'black': [
            (np.array([0, 0, 0]),      np.array([180, 75, 70]))
        ],
        'white': [
            (np.array([0, 0, 200]),    np.array([180, 30, 255]))
        ]
    }

    if target_color not in strict_color_ranges:
        raise ValueError(f"Unsupported color: {target_color}")

    # Build the combined mask
    full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in strict_color_ranges[target_color]:
        sub_mask = cv2.inRange(hsv, lower, upper)
        full_mask = cv2.bitwise_or(full_mask, sub_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

    # Masked image for inspection
    masked_image = cv2.bitwise_and(original, original, mask=full_mask)

    # Find contours
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        print("No valid holds found.")
        empty_grid = np.zeros((40, 26), dtype=np.uint8)
        return [], empty_grid, masked_image, original, (0, 0, original.shape[1], original.shape[0])

    # Determine bounding region that fully contains each hold
    # We'll also check individually if the bounding box is bigger than the current
    # region and expand if needed.
    x_min, y_min, w_min, h_min = cv2.boundingRect(np.vstack(valid_contours))
    x_max = x_min + w_min
    y_max = y_min + h_min

    # We'll expand 15% instead of 10% to ensure we don't clip big holds
    pad_w = int(w_min * 0.15)
    pad_h = int(h_min * 0.15)
    crop_x = max(0, x_min - pad_w)
    crop_y = max(0, y_min - pad_h)
    crop_w = min(original.shape[1] - crop_x, w_min + 2*pad_w)
    crop_h = min(original.shape[0] - crop_y, h_min + 2*pad_h)

    # If any hold's bounding box extends beyond that region, expand further
    # (use the union of each boundingRect).
    for cnt in valid_contours:
        x_i, y_i, w_i, h_i = cv2.boundingRect(cnt)
        if x_i < crop_x:
            dx = crop_x - x_i
            crop_x = x_i
            crop_w += dx
        if y_i < crop_y:
            dy = crop_y - y_i
            crop_y = y_i
            crop_h += dy
        if (x_i + w_i) > (crop_x + crop_w):
            crop_w = (x_i + w_i) - crop_x
        if (y_i + h_i) > (crop_y + crop_h):
            crop_h = (y_i + h_i) - crop_y

    # Ensure crop_x, crop_y in bounds
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    if (crop_x + crop_w) > original.shape[1]:
        crop_w = original.shape[1] - crop_x
    if (crop_y + crop_h) > original.shape[0]:
        crop_h = original.shape[0] - crop_y

    cropped_region = (crop_x, crop_y, crop_w, crop_h)

    # Now build shape-based grid: (40×26)
    crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for cnt in valid_contours:
        # Shift each contour so that (crop_x, crop_y) is origin
        shifted = cnt - [crop_x, crop_y]
        cv2.drawContours(crop_mask, [shifted], -1, 255, thickness=-1)
    
    # Resize to (26 cols, 40 rows)
    grid_cols, grid_rows = 26, 40
    resized_mask = cv2.resize(crop_mask, (grid_cols, grid_rows), interpolation=cv2.INTER_AREA)
    # Binarize
    _, bin_mask = cv2.threshold(resized_mask, 127, 1, cv2.THRESH_BINARY)
    grid_map = bin_mask.astype(np.uint8)

    # Optional morphological close on the grid to fill any holes from resizing
    # If you find it merges adjacent holds too much, remove or reduce kernel size
    kernel_grid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    grid_map = cv2.morphologyEx(grid_map, cv2.MORPH_CLOSE, kernel_grid)

    # Draw bounding boxes & classify
    result_image = original.copy()
    holds_info = []
    for cnt in valid_contours:
        area = cv2.contourArea(cnt)
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
        cx = x_cnt + w_cnt // 2
        cy = y_cnt + h_cnt // 2

        hold_type = classify_hold(cnt, area_adjustment_factor=0.5)

        # Draw
        cv2.rectangle(result_image, (x_cnt, y_cnt),
                      (x_cnt + w_cnt, y_cnt + h_cnt),
                      (0, 255, 0), 2)
        cv2.putText(result_image, hold_type, (x_cnt, y_cnt - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.circle(result_image, (cx, cy), 4, (0,255,255), -1)

        holds_info.append({
            "type": hold_type,
            "position": (cx, cy),
            "bounding_rect": (x_cnt, y_cnt, w_cnt, h_cnt),
            "area": area
        })

    # Also draw the final crop region in red for debugging
    cv2.rectangle(result_image, (crop_x, crop_y),
                  (crop_x+crop_w, crop_y+crop_h),
                  (0,0,255), 2)

    return holds_info, grid_map, masked_image, result_image, cropped_region

def save_results(
    image_path,
    holds_info,
    grid_map,
    result_image,
    masked_image,
    cropped_region,
    output_dir="results"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(f"{output_dir}/{base_name}_masked_{ts}.jpg", masked_image)
    cv2.imwrite(f"{output_dir}/{base_name}_result_{ts}.jpg", result_image)

    # Crop region
    x, y, w, h = cropped_region
    orig = cv2.imread(image_path)
    cropped_img = orig[y:y+h, x:x+w]
    cv2.imwrite(f"{output_dir}/{base_name}_cropped_{ts}.jpg", cropped_img)

    # Save grid
    np.savetxt(f"{output_dir}/{base_name}_grid_{ts}.csv", grid_map, delimiter=",", fmt="%d")

    # Save hold info
    with open(f"{output_dir}/{base_name}_holds_{ts}.txt", "w") as f:
        f.write(f"Detected {len(holds_info)} holds.\n")
        f.write(f"Cropped region: x={x}, y={y}, w={w}, h={h}\n")
        f.write("-"*40 + "\n")
        for i, hold in enumerate(holds_info, 1):
            f.write(f"Hold {i}: {hold['type']}\n")
            f.write(f"  position: {hold['position']}\n")
            f.write(f"  bounding_rect: {hold['bounding_rect']}\n")
            f.write(f"  area: {hold['area']}\n")
            f.write("-"*40 + "\n")

    print(f"Results saved to {output_dir}/")

def main():
    """
    A main function with prompts, detection, shape-based grid generation,
    and final matplotlib display. 
    """
    print("CLIMBING HOLDS DETECTOR - Extended Purple & Better Crop")
    print("="*50)

    # Prompts
    default_image = "climbing_wall.jpg"
    path_in = input(f"Image path (default: {default_image}): ").strip()
    if not path_in:
        path_in = default_image

    color_options = ["red","blue","green","yellow","orange","purple","black","white"]
    print("Colors:", ", ".join(color_options))
    color_in = input("Color to detect (default: purple): ").strip().lower()
    if not color_in or color_in not in color_options:
        color_in = "purple"

    min_area_str = input("Minimum area to be considered a hold (default=200): ").strip()
    min_area = int(min_area_str) if min_area_str.isdigit() else 200

    # Detect
    holds_info, grid_map, masked_image, result_image, cropped_region = detect_and_classify_holds(
        path_in,
        target_color=color_in,
        min_area=min_area
    )

    print(f"Detected {len(holds_info)} holds.")
    for i, hold in enumerate(holds_info, start=1):
        print(f"{i}. {hold}")

    # Show final images
    original = cv2.imread(path_in)
    x, y, w, h = cropped_region
    cropped_img = original[y:y+h, x:x+w]

    plt.figure(figsize=(14,8))
    plt.subplot(2,3,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.title("Masked Image")
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.title("Result with BBoxes")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.title("Cropped Region")
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2,3,(5,6))
    plt.title("Shape-Based Grid (40×26)")
    plt.imshow(grid_map, cmap='binary')
    plt.xticks(range(0, grid_map.shape[1], 1))
    plt.yticks(range(0, grid_map.shape[0], 1))
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # Optionally save
    ans = input("Save results? (y/n, default=y): ").strip().lower()
    if ans != 'n':
        save_results(path_in, holds_info, grid_map, result_image, masked_image, cropped_region)

if __name__ == "__main__":
    main()
