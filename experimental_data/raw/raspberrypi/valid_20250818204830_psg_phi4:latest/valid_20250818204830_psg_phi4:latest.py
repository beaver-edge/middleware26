import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TensorFlow Lite interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape and set the model input size
input_shape = input_details[0]['shape']
interpreter.resize_tensor_input(input_details[0]['index'], [1, *input_shape[1:]])
interpreter.allocate_tensors()  # Reallocate tensors after resizing

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    
    # Normalize to [0, 1] and convert to UINT8
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()

    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int64)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw rectangles and labels on detected objects
    for box, score, class_id in zip(boxes, scores, classes):
        if score >= confidence_threshold:
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = int(xmin * frame_width), int(xmax * frame_width), int(ymin * frame_height), int(ymax * frame_height)
            label = f"{labels[class_id - 1]}: {int(score * 100)}%"
            color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the frame with detections to output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()