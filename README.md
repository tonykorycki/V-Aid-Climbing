# Raspberry Pi Climbing Vision Project

This project is designed to analyze climbing routes using computer vision techniques on a Raspberry Pi 5. It utilizes a YOLO model for detecting holds and volumes in climbing wall images, maps these holds onto a grid, and generates descriptive information using a hosted LLM API on Hugging Face.

## Project Structure

```
pi-climbing-vision
├── src
│   ├── pi_API_test.py          # Main test script for image processing and LLM integration
│   ├── utils
│   │   ├── camera_helper.py     # Functions for camera setup and image capture
│   │   ├── detection.py         # YOLO detection functions for holds and volumes
│   │   ├── grid_mapping.py      # Functions for mapping holds onto a grid
│   │   └── llm_client.py        # Functions for communicating with the LLM API
│   ├── config.py                # Configuration settings for paths and API endpoints
│   └── models
│       └── train4/weights/best.pt # YOLO model weights
├── data
│   ├── images                   # Directory for test images
│   └── results                  # Directory for processed results
├── requirements.txt             # List of required Python packages
├── setup.sh                     # Setup script with virtual environment
└── activate.sh                  # Script to activate the virtual environment
```

## Setup Instructions

1. **Clone the repository** to your Raspberry Pi:
   ```bash
   git clone https://github.com/tonykorycki/V-Aid-Climbing.git
   cd V-Aid-Climbing/pi-climbing-vision

2. **Get a Hugging Face API token**:
   - Create an account at [huggingface.co](https://huggingface.co)
   - Go to Settings → Access Tokens
   - Create a new token with "read" permissions
   - Update the token in config.py

3. **Run the setup script** to install dependencies in a virtual environment:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Prepare images or camera**:
   - Add test images to `data/images/` directory
   - OR ensure your Pi Camera or USB webcam is connected

## Running the Program

1. **Activate the virtual environment**:
   ```bash
   source activate.sh
   ```

2. **Test the API connection** (optional):
   ```bash
   python src/test_api.py
   ```

3. **Run the main program**:
   ```bash
   python src/pi_API_test.py
   ```

4. **Interact with prompts**:
   - Choose between camera or saved images
   - Select whether to use custom settings
   - Choose the color of holds to detect
   - Decide if you want the description read aloud

5. **View results** in the `data/results` directory

## Interactive Options

The program offers several interactive choices:

- **Image Source**: Camera capture or images from directory
- **Camera Type**: Pi Camera or USB webcam (if available)
- **Custom Settings**: Adjust sensitivity, color detection, and minimum hold area
- **Color Selection**: Choose from red, blue, green, yellow, orange, purple, black, or white
- **Display Options**: Show detection results and read descriptions aloud

## Requirements

- Raspberry Pi 5 (recommended) or Pi 4 with at least 4GB RAM
- Pi Camera or USB webcam (optional for new captures)
- Internet connection for Hugging Face API
- Python 3.7+
- Virtual environment (created by setup.sh)

## Performance Notes

- YOLO detection takes 10-30 seconds per image on a Raspberry Pi 5
- The Pi may heat up during intensive processing - cooling is recommended
- Lower resolution images will process faster

---

To check the API token is working, run:
```bash
python src/test_api.py
```

For optimal performance, update the model URL in config.py to:
```python
LLM_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
