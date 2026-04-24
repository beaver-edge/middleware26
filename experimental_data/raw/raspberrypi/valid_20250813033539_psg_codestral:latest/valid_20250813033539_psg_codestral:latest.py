import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Define paths/parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_map_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Load labels from label map path
with open(label_map_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Instantiate interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file for reading
cap = cv2.VideoCapture(input_path)

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame to match input tensor shape and data type
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(input_details[0]['dtype'])

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference on the input tensor
    interpreter.invoke()

    # Retrieve output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret results using label list if provided
    predicted_label = labels[np.argmax(output_data)]

    # Handle output (e.g., display, write to file)
    print('Predicted Label:', predicted_label)

# Release resources acquired during video processing
cap.release()