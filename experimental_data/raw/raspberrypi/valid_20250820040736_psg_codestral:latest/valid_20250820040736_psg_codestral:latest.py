import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Phase 1: Setup
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file for input and get its properties
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
total_frames = int(cap.get(7))

# Define codec and create VideoWriter object for output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

for _ in range(total_frames):
    # Phase 2: Input Acquisition & Preprocessing
    ret, frame = cap.read()
    if not ret:
        break

    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold and scores[0][i] <= 1.0:
            ymin, xmin, ymax, xmax = boxes[0][i]
            label = labels[int(classes[0][i])]
            confidence = int(scores[0][i]*100)

            pt1 = (int(xmin*frame_width), int(ymin*frame_height))
            pt2 = (int(xmax*frame_width), int(ymax*frame_height))

            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), thickness=2)
            text = f"{label}: {confidence}%"
            cv2.putText(frame, text, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)

    out.write(frame)

cap.release()
out.release()