import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define necessary variables and load label file into a Python list
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5
labels = open(label_path).read().strip().split('\n')

# Instantiate the Interpreter and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open input video file and initialize output video writer
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data for inference
    input_data = np.expand_dims(cv2.resize(frame, (300, 300)), axis=0).astype('uint8')

    # Set the tensor to point to the input data to be used for inference
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and text size
            ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
            xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
            ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
            xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))
            (text_width, text_height), baseline = cv2.getTextSize(labels[int(classes[i])], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # Draw box and label for detected object
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.rectangle(frame, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), (10, 255, 0), -1)
            cv2.putText(frame, labels[int(classes[i])], (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Write frame with detections to output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()