# V-Aid Climbing: Vision-Assisted System for Visually Impaired Climbers

![Status: Active](https://img.shields.io/badge/Status-Active-green)
![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue)
![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)

A computer vision system designed to help visually impaired climbers understand climbing routes through detection, tactile feedback, and audio descriptions.

## Table of Contents

- Overview
- Components
- Pi-Climbing-Vision
  - Setup Instructions
  - Running the Program
  - Requirements
  - Performance Notes
- Tactile Plotter System
  - Hardware Components
  - Plotter Features
  - Coordinate System
- Computer Vision Scripts
  - Setup for Computer
  - Running the Computer Scripts
  - Setting Up Local LLM
- Features
- Text-to-Speech Options
- Output
- Advanced Configuration
- Troubleshooting
- License

---

## Repository Structure

```
├── pi-climbing-vision/         # Raspberry Pi optimized code
│   ├── src/                    # Source code
│   │   ├── master.py           # Main entry point with tactile output
│   │   ├── pi_CV_main.py       # Vision + LLM without tactile
│   │   ├── pi_API_test.py      # Simplified API test version
│   │   ├── paths.py            # Configuration paths
│   │   ├── tests/              # Test scripts
│   │   ├── models/             # YOLO models
│   │   └── utils/              # Utility functions
│   ├── data/                   # Data directories
│   │   ├── images/             # Input images
│   │   └── results/            # Output results
│   ├── setup.sh                # Setup script
│   ├── activate.sh             # Environment activation
│   └── run_headless.sh         # Headless execution script
├── computer_vision/            # Computer-based analysis scripts
│   ├── CV_LLM_integration.py   # Optimized with LLM
│   ├── CV-ML-2.py              # ML difficulty prediction
│   ├── CV_type2.py             # Newest CV script with auto-brightness
│   ├── results/                # Output results
│   └── train4/                 # Training configuration
```

## Overview

This project provides computer vision tools to analyze climbing routes by detecting holds, mapping them onto a grid, and generating natural language descriptions. The system uses YOLO object detection to identify climbing holds and volumes, a Language Model (LLM) to describe the routes, and a tactile plotter system to create physical representations for the visually impaired.



## Components

The project has three main components:
1. **Pi-Climbing-Vision**: Optimized vision system for Raspberry Pi devices
2. **Tactile Plotter System**: Hardware setup with Arduino to create physical maps
3. **Computer Vision Scripts**: Can be run on any computer with Python support

---

## Pi-Climbing-Vision

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tonykorycki/V-Aid-Climbing.git
   cd V-Aid-Climbing/pi-climbing-vision
   ```

2. **Get a Hugging Face API token**:
   - Create an account at [huggingface.co](https://huggingface.co)
   - Go to Settings → Access Tokens
   - Create a new token with "read" permissions
   - Update the token in paths.py

3. **Run the setup script** to create a virtual environment and install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Prepare images or camera**:
   - Add test images to `data/images/` directory
   - OR ensure your Pi Camera or USB webcam is connected

5. **Hardware Setup** (for tactile feedback):
   - Connect Arduino with GRBL firmware to Raspberry Pi via USB
   - Ensure the plotter hardware is assembled with servo actuator
   - Connect GPIO buttons (pins 17 and 27) for user interaction

### Running the Program

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
   python src/master.py  # Full integrated version with tactile output
   # OR
   python src/pi_CV_main.py  # Vision and LLM without tactile
   # OR
   python src/pi_API_test.py  # Simplified API test version
   ```

4. **Debug the plotter** (if needed):
   ```bash
   python src/tests/test_plotter.py
   ```

5. **Test TTS engines** (if needed):
   ```bash
   python src/tests/test_tts.py
   ```

6. **Follow the interactive prompts** to:
   - Choose between camera or saved images
   - Select hold colors to detect
   - Configure sensitivity and detection parameters
   - Create tactile representations of routes
   - Generate and hear route descriptions

### Requirements

- Raspberry Pi 5 (recommended) or Pi 4 with at least 4GB RAM
- Pi Camera or USB webcam (optional for new captures)
- Internet connection for Hugging Face API
- Arduino with GRBL firmware for tactile output
- 2-axis plotter setup with servo actuator
- Momentary push buttons for user interface
- Python 3.7+
- Virtual environment (created by setup.sh)

### Performance Notes

- YOLO detection takes 10-30 seconds per image on a Raspberry Pi 5
- The Pi may heat up during processing - cooling is recommended
- Lower resolution images will process faster
- TTS with Google provides better quality but requires internet 
- SVOX Pico TTS provides faster responses for basic UI interactions

---

## Tactile Plotter System

The system includes a 2D plotter with servo actuator that creates tactile representations of detected climbing holds:

### Hardware Components

- Arduino board with GRBL firmware
- 2-axis plotter system (X/Y movement)
- Servo actuator for pressing pins
- Optional: GPIO buttons for hands-free control

### Plotter Features

- Maps climbing holds onto a configurable grid (default 12×12)
- Creates physical map of detected climbing holds
- Each hold is physically represented by actuator movement
- Calibration utilities for precise positioning
- Custom G-code generation based on detected holds
- Offsets configurable for various plotter setups

### Coordinate System

- (0,0) is at the bottom left corner of the grid
- X increases to the right, Y increases upward (in logical coordinates)
- Physical movements are inverted (negative coordinates) to match plotter mechanics

---

## Computer Vision Scripts

You can also run the analysis on a regular computer (not just Raspberry Pi) using the scripts in the computer_vision directory.

### Setup for Computer

1. Install required Python packages:
   ```bash
   pip install numpy opencv-python matplotlib torch ultralytics pillow requests huggingface_hub pyttsx3
   ```

2. (Optional) For local LLM functionality:
   ```bash
   pip install llama-cpp-python huggingface_hub
   ```

### Running the Computer Scripts

You can use any of these three main scripts:

- **CV_LLM_integration.py**: Optimized version with LLM integration
  ```bash
  python computer_vision/CV_LLM_integration.py
  ```
- **CV-ML-2.py**: Version with ML difficulty prediction (no LLM)
  ```bash
  python computer_vision/CV-ML-2.py
  ```
- **CV_type2.py**: Newest CV script with auto-brightness adjustment and LLM integration
  ```bash
  python computer_vision/CV_type2.py
  ```





### Setting Up Local LLM

To use the local LLM functionality (instead of API):

1. When prompted during script execution, choose "y" when asked to install llama-cpp-python and huggingface_hub
2. The script will guide you to select a model from available GGUF models (Llama 2 variants)
3. The model will be downloaded (may be several GB) and tested
4. For subsequent runs, the script will use the downloaded model
5. Select "local" when asked to use local LLM or API

---

## Features

- **Color Detection**: Isolate holds of a specific color (red, blue, green, yellow, etc.)
- **Hold Classification**: Detect and classify both small holds and larger volumes
- **Grid Mapping**: Convert detected holds to a 12x12 grid representation
- **Tactile Output**: Create physical maps using the plotter system
- **Audio Interface**: Voice prompts and speech output for visually impaired users
- **Route Description**: Generate natural language descriptions of the climbing route
- **Text-to-Speech**: Multiple TTS engines for high-quality speech output
- **Button Interface**: Physical buttons for navigation without screen
- **Visualization**: Various visualization options for detection results

## Text-to-Speech Options

The system supports multiple TTS engines:
- **SVOX Pico**: Fast responses, good clarity, works offline
- **Google TTS**: Highest quality, requires internet connection
- **Festival**: Alternative offline option
- **MBROLA**: Enhanced quality offline option with specialized voices
- **pyttsx3**: Default fallback option

## Output

The system produces several outputs:
- Annotated images showing detected holds
- Grid maps representing the climbing route
- Tactile physical representation via the plotter
- Audio descriptions read aloud
- Text descriptions of the route generated by the LLM
- CSV files with grid mapping data
- Text files with details about each detected hold

## Advanced Configuration

- To optimize performance on Raspberry Pi, adjust the YOLO model path in paths.py
- For improved LLM responses, you can try different models in paths.py:
  ```python
  LLM_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
  ```
- To configure the tactile plotter, use `src/test_plotter.py` for calibration

## Troubleshooting

- If camera detection fails, ensure your camera permissions are set correctly
- For memory issues on Raspberry Pi, try reducing image resolution in paths.py
- If LLM responses are slow, consider using the API option instead of local models
- For plotter issues, run `src/tests/test_plotter.py` to test connections and calibration
- If TTS isn't working, run `test_tts.py` to check which engines are available

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

*This project was created to assist visually impaired climbers and is under active development.*