import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# CONFIGURATION PARAMETERS
model_path    = "models/ssd-mobilenet_v1/detect.tflite"
label_path    = "models/ssd-mobilenet_v1/labelmap.txt"
input_path    = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Importing necessary libraries and loading labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))

    # Convert to UINT8 by ensuring pixel values are within 0-255 range
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Phase 3: Inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    h, w = frame.shape[:2]
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = int(classes[i]) - 1
            label = labels[class_id]

            # Extract bounding box coordinates
            box = boxes[i][0:4]
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = (xmin * w, xmax * w, ymin * h, ymax * h)

            left, right, top, bottom = int(left), int(right), int(top), int(bottom)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            text = f"{label}: {scores[i]:.2f}"
            cv2.putText(frame, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out.write(frame)

# Phase 4: Output Interpretation & Handling Loop
cap.release()
out.release()

print("Video processing complete and saved to:", output_path)