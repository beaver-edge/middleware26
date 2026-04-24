import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define necessary variables
model_path    = "models/ssd-mobilenet_v1/detect.tflite"
label_path    = "models/ssd-mobilenet_v1/labelmap.txt"
input_path    = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Acquire Input Data
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = 300 # fix the width to match model's expected input shape
height = 300 # fix the height to match model's expected input shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    frame = cv2.resize(frame, (width, height)) # resize the frame to match model's expected input shape
    input_data = np.expand_dims(frame, axis=0)
    input_data = np.uint8((input_data / 255.0) * 255.0)

    # Set Input Tensor and Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Output Tensor and Interpret Results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            label_text = labels[classes[i]] + ": " + str(round(scores[i], 2))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (xmin + 10, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detections into the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()