import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Define variables
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
labelmap_path = 'models/ssd-mobilenet_v1/labelmap.txt'
confidence_threshold = 0.5
input_video_path = 'data/object_detection/sheeps.mp4'
output_video_path = 'results/object_detection/test_results/sheeps_detections.mp4'

# Load labels
with open(labelmap_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter and get model details
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Open video file for input and output
video_in = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# Input acquisition and preprocessing loop
while video_in.isOpened():
    ret, frame = video_in.read()
    if not ret:
        break

    # Preprocess data
    input_data = np.expand_dims(cv2.resize(frame, (width, height)), axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output and interpret results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int((boxes[i][0]*height))
            xmin = int((boxes[i][1]*width))
            ymax = int((boxes[i][2]*height))
            xmax = int((boxes[i][3]*width))

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, ymin - labelSize[1] - 10), (xmin + labelSize[0], ymin + baseLine - 10), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video file
    video_out.write(frame)

# Cleanup
video_in.release()
video_out.release()
cv2.destroyAllWindows()