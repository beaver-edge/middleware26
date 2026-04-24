import numpy as np
import os
import time
from ai_edge_litert.interpreter import Interpreter
import cv2

# Phase 1: Setup
## 1.1 Imports: Already done above

## 1.2 Paths/Parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_map_path = 'models/ssd-mobilenet_v1/labelmap.txt'
confidence_threshold = 0.5
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'

## 1.3 Load Labels (Conditional)
if os.path.exists(label_map_path):
    with open(label_map_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
else:
    labels = []

## 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

## 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
## 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

## 2.2 Preprocess Data
input_shape = input_details[0]['shape']
preprocessed_frames = []

# Phase 3: Inference (Run per preprocessed input)
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_shape[1], input_shape[2]))
    return np.expand_dims(frame_resized, axis=0).astype(input_details[0]['dtype'])

# Phase 4: Output Interpretation & Handling Loop
## 4.1 Get Output Tensor(s)
def postprocess_output(output_data):
    detection_boxes = output_data[0][0]
    detection_classes = output_data[1][0].astype(int)
    detection_scores = output_data[2][0]
    
    results = []
    for i in range(len(detection_scores)):
        if detection_scores[i] > confidence_threshold:
            class_id = detection_classes[i]
            box = detection_boxes[i]
            label = labels[class_id] if class_id < len(labels) else f'unknown_{class_id}'
            score = detection_scores[i]
            results.append((box, label, score))
    return results

## 4.2 Interpret Results
def draw_results(frame, results):
    for box, label, score in results:
        ymin, xmin, ymax, xmax = box
        height, width, _ = frame.shape
        (xminn, xmaxx, yminn, ymaxx) = (int(xmin * width), int(xmax * width),
                                        int(ymin * height), int(ymax * height))
        
        cv2.rectangle(frame, (xminn, yminn), (xmaxx, ymaxx), (0, 255, 0), 2)
        label_text = f'{label}: {score:.2f}'
        cv2.putText(frame, label_text, (xminn, yminn - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

## 4.3 Handle Output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    results = postprocess_output(output_data)
    draw_results(frame, results)
    
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()