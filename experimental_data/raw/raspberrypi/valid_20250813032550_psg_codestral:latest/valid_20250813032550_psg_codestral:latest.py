import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Define paths/parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
video_path = 'path_to_your_video.mp4'
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels (if provided and relevant)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = "data/object_detection/sheeps.mp4"
output_details = interpreter.get_output_details()

# Open video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Acquire input data
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data (resize and normalize for example)
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0

    # Set input tensor(s) and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret results (use labels if relevant)
    result = np.argmax(output_data, axis=1)[0]
    label = labels[result]

    # Handle output (print or write to file for example)
    print('Detected:', label)

# Release resources
cap.release()