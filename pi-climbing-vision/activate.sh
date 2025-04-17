#!/bin/bash
source .venv/bin/activate
echo "Virtual environment activated. Run 'deactivate' when finished."
echo "Run 'python src/pi_CV_main.py' to start the application."
echo " To test components:"
echo "   - API: python src/test_api.py"
echo "   - Audio: python src/tests/test_audio.py"
echo "   - Text-to-speech: python src/tests/test_tts.py"
echo "   - Buttons: python src/tests/test_buttons.py"
echo "   - Arduino: python src/tests/test_arduino.py"