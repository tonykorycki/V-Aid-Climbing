import cv2
import numpy as np

def classify_hold(cnt):
    """ Classify hold based on shape features """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Aspect ratio (width/height)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    # Convexity (higher = smoother)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0

    # Approximate polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Heuristic Classification
    if aspect_ratio > 1.2 and convexity > 0.8:
        return "jug"
    elif len(approx) > 10 and convexity < 0.75:
        return "sloper"
    elif aspect_ratio < 1.2 and convexity < 0.9:
        return "crimp"
    else:
        return "foothold"

def detect_and_classify_holds(image_path):
    """ Detect holds and classify their types """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for holds (adjust based on actual colors)
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([10, 255, 255])

    # Create a mask for holds
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holds = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        hold_type = classify_hold(cnt)
        holds.append({"type": hold_type, "position": (x + w//2, y + h//2)})

        # Draw bounding box and label
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, hold_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display image with classifications
    cv2.imshow("Detected Holds", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return holds

# Run detection
holds = detect_and_classify_holds("climbing_wall.jpg")
print("Detected Holds:", holds)
