#!/bin/bash

echo "Setting up Pi-Climbing-Vision..."

# Update package list and upgrade installed packages
sudo apt update && sudo apt upgrade -y

# Install required dependencies
sudo apt install -y python3-pip python3-opencv libopencv-dev
pip3 install numpy matplotlib torch torchvision ultralytics pillow requests huggingface_hub pyttsx3

# For Pi Camera setup (if using Pi Camera)
sudo apt install -y python3-picamera2

# Create necessary directories if they don't exist
mkdir -p data/images
mkdir -p data/results

echo "Setup complete. You can now add images to the 'data/images' directory and run the project."