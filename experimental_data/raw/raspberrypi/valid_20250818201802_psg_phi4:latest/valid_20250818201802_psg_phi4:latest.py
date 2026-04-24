import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
def load_labels(label_path):
    with open(label_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

labels = load_labels(label_path)
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Pre-process input data
def preprocess_frame(frame, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
    return np.expand_dims(frame_resized, axis=0).astype(np.uint8)

# Phase 3: Inference function
def run_inference(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    return boxes, classes, scores

# Phase 4: Output interpretation & handling loop
def draw_boxes(frame, boxes, classes, scores):
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Phase 2: Input acquisition & preprocessing loop
cap = cv2.VideoCapture(input_path)
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess_frame(frame, input_details[0]['shape'][1:3])
    boxes, classes, scores = run_inference(input_data)
    draw_boxes(frame, boxes, classes, scores)

    output_video.write(frame)

cap.release()
output_video.release()

# Cleanup resources
cv2.destroyAllWindows()