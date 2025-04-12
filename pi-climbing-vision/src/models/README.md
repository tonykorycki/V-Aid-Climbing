# Models Documentation

## YOLO Model
The project utilizes the YOLO (You Only Look Once) model for real-time object detection. This model is trained to identify climbing holds and volumes in images captured from climbing walls. 

### Model Details
- **Model Type**: YOLOv5 (or specify the version used)
- **Training Data**: The model is trained on a dataset of climbing holds, ensuring it can accurately detect various types of holds in different lighting and background conditions.
- **Output Classes**: The model outputs bounding boxes for detected holds and volumes, along with confidence scores.

### Usage
To use the YOLO model in this project:
1. Ensure the model weights are placed in the directory specified in `config.py`.
2. The model is loaded and utilized in the `src/utils/detection.py` file, where it processes images to identify holds.

## LLM Integration
The project also integrates with a hosted LLM (Language Model) API on Hugging Face to generate descriptions of the climbing routes based on the detected holds.

### API Details
- **API Endpoint**: The endpoint for the LLM API is configured in `src/config.py`.
- **Request Format**: The requests to the API include the grid representation of the detected holds and any additional context needed for generating a description.

### Usage
To generate descriptions:
1. After detecting holds using the YOLO model, the grid representation is created.
2. A request is sent to the LLM API using the functions defined in `src/utils/llm_client.py`.
3. The generated description is then saved in the results directory.

## Additional Information
For further details on the implementation and usage of the models, refer to the respective utility files in the `src/utils` directory.