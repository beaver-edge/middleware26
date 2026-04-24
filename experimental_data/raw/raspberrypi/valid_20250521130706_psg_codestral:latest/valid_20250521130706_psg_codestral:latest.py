import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define paths/parameters
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
labelmap_path = 'models/ssd-mobilenet_v1/labelmap.txt'  # Only if relevant and provided
input_video_path = 'data/object_detection/sheeps.mp4'  # Only if relevant and provided
output_video_path = 'results/object_detection/test_results/sheeps_detections.mp4'
confidence_threshold = 0.5

# Load labels (conditional)
with open(labelmap_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input acquisition & preprocessing loop
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_data = np.expand_dims(cv2.resize(frame, (300, 300)), axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output interpretation & handling loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index number of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int((boxes[i][0] * frame_height))
            xmin = int((boxes[i][1] * frame_width))
            ymax = int((boxes[i][2] * frame_height))
            xmax = int((boxes[i][3] * frame_width))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]  # Map class index to object name
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ymin = max(ymin, labelSize[1])
            cv2.rectangle(frame, (xmin, ymin - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()