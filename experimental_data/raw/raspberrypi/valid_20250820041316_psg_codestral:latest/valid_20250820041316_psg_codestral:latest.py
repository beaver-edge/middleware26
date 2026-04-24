# Import necessary libraries
import numpy as np
from PIL import Image
import cv2
from tflite_runtime.interpreter import Interpreter

# Define variables
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()  # Corrected line

# Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    normalized_input_data = (np.array(input_data) / 127.5).astype('uint8')  # Converted to uint8

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], normalized_input_data)
    interpreter.invoke()

    # Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret Results & Handle Output
    detections = np.squeeze(output_data)
    for i in range(len(detections)):
        if detections[i, 2] > 0.5:
            class_id = int(detections[i, 1])
            label = labels[class_id]
            # Here you can add your code to handle the detection results

# Cleanup
cap.release()