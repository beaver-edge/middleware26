import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2
import time
import os

# Phase 1: Setup
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_map_path = 'models/ssd-mobilenet_v1/labelmap.txt'
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels
with open(label_map_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter and get model details
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Open video file for reading and writing output
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Phase 2: Input Acquisition & Preprocessing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 2.2: Preprocess Data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    # Phase 4.1: Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Phase 4.2: Interpret Results & Handle Output
    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
            xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
            ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
            xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = '{}: {:.2f}'.format(labels[int(classes[i])], scores[i])
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video file
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()