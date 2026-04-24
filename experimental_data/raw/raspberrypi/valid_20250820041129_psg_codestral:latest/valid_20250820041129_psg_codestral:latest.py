import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define paths and parameters
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
confidence_threshold   = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Initialize video capture and output writer
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input data
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)  # Change this line

    # Perform object detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw rectangles around detected objects and add text labels
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            label = labels[int(classes[i])]
            ymin, xmin, ymax, xmax = boxes[i]
            pt1 = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            pt2 = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write the frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()