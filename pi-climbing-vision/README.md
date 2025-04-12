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
│       └── README.md            # Documentation about the models used in the project
├── data
│   ├── images
│   │   └── README.md            # Information about the images directory
│   └── results
│       └── README.md            # Information about the results directory
├── requirements.txt             # List of required Python packages and dependencies
├── setup.sh                     # Shell script for setting up the environment
└── README.md                    # Project documentation and usage instructions
```

## Setup Instructions

1. **Clone the repository** or create the project structure as outlined above.
2. **Place your YOLO model** in the appropriate directory as specified in `config.py`.
3. **Add images** to the `data/images` directory for processing.
4. **Update the `requirements.txt`** file with any additional dependencies you may need.
5. **Run the `setup.sh` script** to install the required packages and set up the environment.
6. **Execute the `src/pi_API_test.py` script** to start the image processing and LLM description generation.
7. **Check the `data/results` directory** for the output results and generated descriptions.

# On the Pi
git clone <your-repository-url>
cd pi-climbing-vision
chmod +x setup.sh
./setup.sh

# Edit configuration
nano src/config.py
# Replace HF_API_TOKEN with your actual token

# Run the program
python3 src/pi_API_test.py

## Usage

- Ensure your Raspberry Pi is set up with the necessary dependencies.
- Use the provided scripts to capture images or use existing images in the `data/images` directory.
- The results will be saved in the `data/results` directory, including the generated descriptions from the LLM.

## Notes

- Performance may vary based on the complexity of the images and the capabilities of the Raspberry Pi.
- Consider using a lighter YOLO model for improved performance if necessary.
- Ensure that your Raspberry Pi has adequate cooling, especially during intensive processing tasks.