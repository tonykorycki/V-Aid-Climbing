#!/bin/bash

echo "Setting up Pi-Climbing-Vision..."

# Update package list ONLY (no upgrade)
sudo apt update

# Install ONLY the required system dependencies
sudo apt install -y python3-pip python3-venv python3-opencv libopencv-dev
sudo apt install -y python3-picamera2  # Pi Camera support

# Create and activate virtual environment
python3 -m venv .venv
echo "Creating virtual environment..."
source .venv/bin/activate

# Install Python dependencies in the virtual environment
pip install --upgrade pip
pip install numpy matplotlib pillow requests huggingface_hub pyttsx3
pip install opencv-python
pip install ultralytics  # This will handle torch installation

# Create necessary directories
mkdir -p data/images
mkdir -p data/results

echo "Setup complete!"
echo "To use the project:"
echo "1. Run 'source activate.sh' to activate the environment"
echo "2. Run 'python src/pi_API_test.py' to start the application"