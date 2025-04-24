# Configuration settings for the Pi Climbing Vision project

# Paths for the YOLO model and image directories
YOLO_MODEL_PATH = "src/models/train4/weights/best.pt" 
IMAGE_DIR = "data/images/"
RESULTS_DIR = "data/results/"

# API endpoint for the hosted LLM on Hugging Face
LLM_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Hugging Face API Token (you need to get this from huggingface.co)
HF_API_TOKEN = "YOUR_TOKEN"

# Detection settings
SENSITIVITY_LEVEL = 25  # Default sensitivity for color detection
MIN_HOLD_AREA = 30     # Minimum area to consider as a hold in pixels

# Camera settings
USE_PI_CAMERA = True
RESOLUTION = (640, 480)