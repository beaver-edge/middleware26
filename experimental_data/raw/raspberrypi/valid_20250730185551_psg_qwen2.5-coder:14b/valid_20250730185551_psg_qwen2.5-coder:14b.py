import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels if provided and relevant
if label_path:
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
import cv2

# Open the video file
cap = cv2.VideoCapture(input_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess the frame
    input_shape = input_details[0]['shape'][1:3]
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb, axis=0).astype(input_details[0]['dtype'])

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if labels:
        # Assuming the model outputs a class index
        predicted_class_index = np.argmax(output_data[0])
        predicted_label = labels[predicted_class_index]
        print(f"Predicted Label: {predicted_label}")

    # Optionally, save or process the output further

# Phase 5: Cleanup
cap.release()