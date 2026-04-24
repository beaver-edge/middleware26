import numpy as np
import time
import os
import cv2

# Phase 1: Setup
model_path     = "models/ssd-mobilenet_v1/detect.tflite"
label_path     = "models/ssd-mobilenet_v1/labelmap.txt"
input_path     = "data/object_detection/sheeps.mp4"
output_path     = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold     = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError("Error opening video file")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess Data
    input_data = np.expand_dims(cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2])), axis=0)
    input_data = input_data.astype(np.uint8)  # Convert to UINT8

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detection
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence score

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_name = labels[int(classes[i])]
            box = boxes[i]
            y1, x1, y2, x2 = int(box[0] * frame.shape[0]), int(box[1] * frame.shape[1]), int(box[2] * frame.shape[0]), int(box[3] * frame.shape[1])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {scores[i]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()