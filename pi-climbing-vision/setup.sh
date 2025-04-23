#!/bin/bash

echo "Setting up Pi-Climbing-Vision..."

# Function to check if a command succeeded
check_status() {
  if [ $? -ne 0 ]; then
    echo "❌ Error: $1 failed. Please check the output above."
    exit 1
  else
    echo "✅ $1 completed successfully."
  fi
}

# Update package list ONLY (no upgrade)
echo "Updating package lists..."
sudo apt update
check_status "Package list update"

# Install ONLY the required system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-venv python3-opencv libopencv-dev
check_status "Core dependencies installation"
sudo apt install -y python3-picamera2  # Pi Camera support
check_status "Pi Camera support installation"

# Install all required X and Qt dependencies for visualization
echo "Installing visualization dependencies..."
sudo apt-get install -y libxcb-xinerama0 libxcb-image0 libxcb-icccm4 libxcb-keysyms1
sudo apt-get install -y libxcb-randr0-dev libxcb-xkb-dev libxcb-icccm4-dev 
sudo apt-get install -y libxcb-image0-dev libxcb-render-util0-dev libxcb1-dev
check_status "Visualization dependencies installation"

# Install virtual framebuffer for headless operation
sudo apt-get install -y xvfb
check_status "Headless operation support installation"

# Install eSpeak specifically - required for pyttsx3 on Linux
echo "Installing eSpeak speech engine..."
sudo apt install -y espeak espeak-ng
sudo apt install -y python3-espeak
check_status "Speech engine installation"

# Fix permissions for audio access
sudo usermod -a -G audio $USER
check_status "Audio permissions setup"

# Install audio dependencies
echo "Installing audio dependencies..."
sudo apt install -y alsa-utils pulseaudio pulseaudio-utils
check_status "Audio dependencies installation"

# For Bluetooth audio support
sudo apt install -y bluez bluez-tools pulseaudio-module-bluetooth
check_status "Bluetooth audio support installation"

# Set audio to force 3.5mm jack output (comment out if using HDMI)
sudo amixer cset numid=3 1

# Quick audio test
echo "Testing audio output..."
speaker-test -t sine -f 440 -c 2 -s 1

# Add to setup.sh for serial port access
sudo usermod -a -G dialout $USER
check_status "Serial port access setup"

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv --system-site-packages
check_status "Virtual environment creation"
source .venv/bin/activate

# Install Python dependencies in the virtual environment
echo "Installing Python packages..."
pip install --upgrade pip
pip install picamera2 numpy matplotlib pillow requests huggingface_hub pyttsx3
pip install opencv-python pyserial
pip install ultralytics  # This will handle torch installation
check_status "Python package installation"

# Add better TTS options confirmed to work on Pi 5
echo "Installing multiple high-quality TTS engines..."

# Install SVOX Pico TTS (Android's TTS engine - good quality)
echo "Installing SVOX Pico TTS..."
sudo apt-get install -y libttspico-utils
check_status "SVOX Pico TTS installation"

# Install Festival TTS (another good quality option)
echo "Installing Festival TTS..."
sudo apt-get install -y festival festvox-us-slt-hts
check_status "Festival TTS installation"

# Install eSpeak-NG with MBROLA voices for better quality
echo "Installing enhanced eSpeak-NG with MBROLA..."
sudo apt-get install -y espeak-ng mbrola
# Download US English voices that are confirmed to work on Pi 5
sudo apt-get install -y mbrola-us1 mbrola-us2 mbrola-us3 || echo "⚠️ MBROLA voices not available, using standard voices"

# Google TTS option (requires internet)
echo "Installing gTTS (Google Text-to-Speech)..."
pip install gtts
sudo apt-get install -y mpg321  # MP3 player for gTTS output
check_status "Google TTS installation"

# Install pygame for audio playback
pip install pygame
check_status "Pygame installation"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/images
mkdir -p data/results
mkdir -p src/models/train4/weights
check_status "Directory creation"

# Check for YOLO model file
if [ ! -f "src/models/train4/weights/best.pt" ]; then
  echo "⚠️ YOLO model weights not found at src/models/train4/weights/best.pt"
  echo "   You will need to download or train the model before using the system."
  echo "   Please see the README.md for instructions."
fi

# Create a convenience script to run with virtual framebuffer if needed
cat > run_headless.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
xvfb-run -a python src/pi_CV_main.py "$@"
EOF
chmod +x run_headless.sh
check_status "Headless script creation"

# Create activation script with helpful messages
cat > activate.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "Virtual environment activated. Run 'deactivate' when finished."
echo "Run 'python src/master.py' to start the application."
echo " To test components:"
echo "   - API: python src/test_api.py"
echo "   - Audio: python src/tests/test_audio.py"
echo "   - Text-to-speech: python src/tests/test_tts.py"
echo "   - Buttons: python src/tests/test_buttons.py"
echo "   - Arduino: python src/tests/test_arduino.py"
echo "   - Camera: python src/tests/test_camera.py"
echo "   - Plotter: python src/tests/test_plotter.py"
EOF
chmod +x activate.sh
check_status "Activation script creation"

# Check if paths.py has a real token
if grep -q "YOUR_TOKEN" "src/paths.py"; then
  echo "⚠️ Remember to set your Hugging Face API token in src/paths.py"
  echo "   You will need this token to use the LLM features."
fi

echo ""
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
echo "   - Plotter: python src/tests/test_plotter.py"
echo "3. Run 'python src/master.py' to start the application"
echo "4. Or run './run_headless.sh' to run in headless mode"