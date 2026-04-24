import numpy as np
from tflite_runtime.interpreter import Interpreter

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels (if provided and relevant)
labels = []
with open(label_path, 'r') as f:
    for line in f:
        labels.append(line.strip())

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
# For video processing, you would typically use OpenCV to read frames.
# Here, we assume a placeholder for input data acquisition and preprocessing.

# Example: Read a single frame from a video file (replace with actual video processing logic)
import cv2

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read frame")

# Preprocess the frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
input_data = np.expand_dims(frame_resized, axis=0).astype(input_details[0]['dtype'])

# Phase 3: Inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Phase 4: Output Interpretation & Handling Loop
output_data = interpreter.get_tensor(output_details[0]['index'])

# Example interpretation (assuming classification task)
if len(output_data.shape) == 2:
    # Softmax the output if necessary
    output_data = np.squeeze(output_data)
    predicted_class_index = np.argmax(output_data)
    predicted_class_label = labels[predicted_class_index]
    confidence = output_data[predicted_class_index]

    print(f"Predicted Class: {predicted_class_label}, Confidence: {confidence:.2f}")

# Phase 5: Cleanup
cap.release()