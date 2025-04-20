#!/bin/bash

echo "Setting up Pi-Climbing-Vision..."

# Update package list ONLY (no upgrade)
sudo apt update

# Install ONLY the required system dependencies
sudo apt install -y python3-pip python3-venv python3-opencv libopencv-dev
sudo apt install -y python3-picamera2  # Pi Camera support

# Install all required X and Qt dependencies for visualization
sudo apt-get install -y libxcb-xinerama0 libxcb-image0 libxcb-icccm4 libxcb-keysyms1
# Additional X11/Qt dependencies to prevent visualization issues
sudo apt-get install -y libxcb-randr0-dev libxcb-xkb-dev libxcb-icccm4-dev 
sudo apt-get install -y libxcb-image0-dev libxcb-render-util0-dev libxcb1-dev
# Install dependencies
sudo apt-get install -y python3-pip portaudio19-dev python3-pygame
# Install virtual framebuffer for headless operation
sudo apt-get install -y xvfb

# Install eSpeak specifically - required for pyttsx3 on Linux
echo "Installing eSpeak speech engine..."
sudo apt install -y espeak espeak-ng
sudo apt install -y python3-espeak

# Add this section before the pip installs:

echo "Installing Mimic3 TTS via apt..."
sudo apt-get install -y mimic3
# Then install Python bindings with pip if available
pip install mycroft-mimic3-tts || echo "Python bindings not available via pip, using system installation"

# Fix permissions for audio access
sudo usermod -a -G audio $USER

# Install audio dependencies
echo "Installing audio dependencies..."
sudo apt install -y alsa-utils pulseaudio pulseaudio-utils

# For Bluetooth audio support
sudo apt install -y bluez bluez-tools pulseaudio-module-bluetooth

# Set audio to force 3.5mm jack output (comment out if using HDMI)
sudo amixer cset numid=3 1

# Quick audio test
echo "Testing audio output..."
speaker-test -t sine -f 440 -c 2 -s 1

# Add to setup.sh for serial port access
sudo usermod -a -G dialout $USER

# Create and activate virtual environment
python3 -m venv .venv --system-site-packages
echo "Creating virtual environment..."
source .venv/bin/activate

# Install Python dependencies in the virtual environment
pip install --upgrade pip
pip install picamera2
pip install numpy matplotlib pillow requests huggingface_hub pyttsx3
pip install opencv-python
pip install pyserial
pip install ultralytics  # This will handle torch installation
#pip install mimic3-tts pygame

pip install pygame


# Create necessary directories
mkdir -p data/images
mkdir -p data/results

# Create a convenience script to run with virtual framebuffer if needed
cat > run_headless.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
xvfb-run -a python src/pi_CV_main.py "$@"
EOF
chmod +x run_headless.sh

# Add after TTS installation:

echo "Testing TTS system..."
if command -v mimic3 &> /dev/null; then
    mimic3 "TTS installation successful" | aplay
    echo "✓ Mimic3 TTS working!"
else
    echo "⚠️ Mimic3 not available in path, check installation"
fi

echo "===== Setup complete! ====="
echo "To use the project:"
echo "1. Run 'source activate.sh' to activate the environment"
echo "2. To test components: (optional)"
echo "   - API: python src/test_api.py"
echo "   - Audio: python src/tests/test_audio.py"
echo "   - Text-to-speech: python src/tests/test_tts.py"
echo "   - Buttons: python src/tests/test_buttons.py"
echo "   - Arduino: python src/tests/test_arduino.py"
echo "   - Camera: python src/tests/test_camera.py"
echo "3. Run 'python src/master.py' to start the application"
echo "4. Or run './run_headless.sh' to run in headless mode"