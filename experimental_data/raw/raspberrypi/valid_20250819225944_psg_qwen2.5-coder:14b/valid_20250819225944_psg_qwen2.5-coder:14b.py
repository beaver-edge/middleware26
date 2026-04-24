import os
import numpy as np
import cv2

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter
from tflite_runtime.interpreter import Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_tensor = np.expand_dims(cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1])), axis=0)
    input_tensor = input_tensor.astype(np.uint8)  # Convert to UINT8
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Phase 3: Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # Get output tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Assuming output is in [1, num_boxes, 4]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Assuming output is in [1, num_boxes]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Assuming output is in [1, num_boxes]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = int(classes[i])
            label = labels[class_id] if class_id < len(labels) else 'unknown'
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            y_min, x_min, y_max, x_max = int(y_min * frame.shape[0]), int(x_min * frame.shape[1]), int(y_max * frame.shape[0]), int(x_max * frame.shape[1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()