import cv2
import numpy as np

# Configuration parameters from the prompt
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Open video file and prepare output writer
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Phase 3: Inference per frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input data
    input_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    
    # Normalize the image to [0, 1] and convert to UINT8
    input_data = np.expand_dims(input_frame / 255.0, axis=0).astype(np.uint8)

    # Set input tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor(s)
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    detection_classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    detection_scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Phase 4: Interpret Results and Handle Output
    for i in range(len(detection_classes)):
        if detection_scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(detection_classes[i])]
            label = f'{object_name}: {detection_scores[i]:.2f}'
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detections to the output file
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()