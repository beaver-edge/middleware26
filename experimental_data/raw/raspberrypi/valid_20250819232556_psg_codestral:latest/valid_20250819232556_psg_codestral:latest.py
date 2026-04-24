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

# Load interpreter and get input/output details
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video capture and output video writer
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data: resize and normalize the input image
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.uint8(input_data)  # fix the type error

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor and interpret results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Handle output: draw bounding boxes on the frame and write it to the output video
    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * frame_height)))
            xmin = int(max(1,(boxes[i][1] * frame_width)))
            ymax = int(min(frame_height,(boxes[i][2] * frame_height)))
            xmax = int(min(frame_width,(boxes[i][3] * frame_width)))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            label = '{}: {:.2f}'.format(labels[int(classes[i])], scores[i])
            cv2.putText(frame, label, (xmin, min(ymin, frame_height - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out.write(frame)

# Release resources
cap.release()
out.release()