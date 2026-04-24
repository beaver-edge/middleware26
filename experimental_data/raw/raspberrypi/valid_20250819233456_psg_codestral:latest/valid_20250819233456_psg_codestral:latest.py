import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Paths and parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to draw bounding boxes on the frame
def draw_boxes(frame, boxes, classes, scores, labels, confidence_threshold):
    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            # Get the box coordinates
            ymin = int(max(1,(boxes[i][0] * frame.shape[0])))
            xmin = int(max(1,(boxes[i][1] * frame.shape[1])))
            ymax = int(min(frame.shape[0],(boxes[i][2] * frame.shape[0])))
            xmax = int(min(frame.shape[1],(boxes[i][3] * frame.shape[1])))

            # Draw the box and label on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = '{}: {:.2f}'.format(labels[int(classes[i])], scores[i])
            cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Input acquisition and preprocessing loop
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the frame to match the expected input size
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(resized_frame, axis=0).astype(input_details[0]['dtype'])

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output interpretation and handling loop
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Draw bounding boxes on the frame and write to output video file
    output_frame = draw_boxes(frame, boxes[0], classes[0], scores[0], labels, confidence_threshold)
    out.write(output_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()