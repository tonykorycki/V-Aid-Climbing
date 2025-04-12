# Instructions for the images directory

This directory is intended for storing input images that will be processed by the climbing route analyzer. 

To use this directory:

1. Place your images in this folder. The images should be in a format supported by OpenCV (e.g., JPG, PNG).
2. Ensure that the images are clear and well-lit to improve detection accuracy.
3. You can add multiple images for batch processing.

After adding images, you can run the main script located in `src/pi_API_test.py` to analyze the climbing holds and generate descriptions using the LLM API. The results will be saved in the `data/results` directory.