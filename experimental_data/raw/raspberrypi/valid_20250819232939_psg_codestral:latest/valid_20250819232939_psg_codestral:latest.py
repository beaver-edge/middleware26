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

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open input video
cap = cv2.VideoCapture(input_path)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess input data
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(resized_frame, axis=0).astype(input_details[0]['dtype'])

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Interpret results and handle output
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = int(classes[i])
            label = labels[class_id]
            ymin, xmin, ymax, xmax = boxes[i]
            height, width = frame.shape[:2]
            x1 = int(xmin * width)
            y1 = int(ymin * height)
            x2 = int(xmax * width)
            y2 = int(ymax * height)

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()