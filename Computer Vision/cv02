import cv2
import numpy as np
import time
from gtts import gTTS
import os

# Global variable for selected color
selected_color = None

# Color dictionary for HSV ranges
COLOR_RANGES = {
    "red": ([0, 100, 100], [10, 255, 255]),
    "blue": ([100, 150, 0], [140, 255, 255]),
    "green": ([40, 70, 70], [80, 255, 255]),
    "yellow": ([20, 100, 100], [40, 255, 255])
}

# Function to give audio feedback
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("feedback.mp3")
    os.system("mpg321 feedback.mp3")

# Function to classify hold type
def classify_hold(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area != 0 else 0
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    if aspect_ratio > 1.2 and convexity > 0.8:
        return "jug"
    elif len(approx) > 10 and convexity < 0.75:
        return "sloper"
    elif aspect_ratio < 1.2 and convexity < 0.9:
        return "crimp"
    else:
        return "foothold"

# Function to detect holds
def detect_and_classify_holds(image_path, color):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = map(np.array, COLOR_RANGES[color])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holds = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        hold_type = classify_hold(cnt)
        holds.append({"type": hold_type, "position": (x + w//2, y + h//2)})
    
    return holds

# Function to cycle through colors
def select_color():
    global selected_color
    colors = list(COLOR_RANGES.keys())
    index = 0
    
    while True:
        speak(f"Selected color: {colors[index]}")
        time.sleep(1)
        user_input = input("Press 'n' for next color, 's' to select: ")
        
        if user_input == 'n':
            index = (index + 1) % len(colors)
        elif user_input == 's':
            selected_color = colors[index]
            speak(f"Color {selected_color} selected.")
            break
    
    return selected_color

if __name__ == "__main__":
    color = select_color()
    holds = detect_and_classify_holds("climbing_wall.jpg", color)
    print("Detected Holds:", holds)
