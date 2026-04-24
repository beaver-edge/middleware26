from ai_edge_litert.interpreter import Interpreter
import numpy as np

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels if provided and relevant
labels = []
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

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = input_data.astype(input_details[0]['dtype'])
    input_data = np.expand_dims(input_data, axis=0)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assuming the model outputs class probabilities for each label
    predicted_class_index = np.argmax(output_data)
    predicted_label = labels[predicted_class_index] if labels else f"Class {predicted_class_index}"

    print(f"Predicted Label: {predicted_label}")

# Phase 5: Cleanup
cap.release()