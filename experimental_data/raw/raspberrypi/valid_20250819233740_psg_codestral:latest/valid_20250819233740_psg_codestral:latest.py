import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define paths and parameters
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter and get model details
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video capture and output writer
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = np.expand_dims(cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2])), axis=0)

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor and interpret results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0].astype('int')
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Filter detections based on confidence threshold
    valid_indices = np.where(scores > confidence_threshold)[0]
    classes = [labels[i] for i in class_ids[valid_indices]]

    # Draw bounding boxes and labels on frame
    for box, cls in zip(boxes[valid_indices], classes):
        ymin, xmin, ymax, xmax = box
        h, w, _ = frame.shape
        p1 = (int(xmin * w), int(ymin * h))
        p2 = (int(xmax * w), int(ymax * h))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, cls, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()