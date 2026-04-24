import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Paths/Parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_map_path = 'models/ssd-mobilenet_v1/labelmap.txt'
confidence_threshold = 0.5
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'

# Load Labels
with open(label_map_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video capture and output writer
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Input Acquisition & Preprocessing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)

    # Set Input Tensor(s) and Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Interpret Results & Handle Output
    for i in range(num_detections):
        if scores[0, i] > confidence_threshold:
            class_id = int(classes[0, i])
            label = labels[class_id]
            ymin, xmin, ymax, xmax = boxes[0, i]
            height, width = frame.shape[:2]
            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()