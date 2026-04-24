import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define necessary variables and paths
model_path    = "models/ssd-mobilenet_v1/detect.tflite"
label_path    = "models/ssd-mobilenet_v1/labelmap.txt"
input_path    = "data/object_detection/sheeps.mp4"
output_path    = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold    = 0.5

# Load Labels
with open(label_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = np.expand_dims(cv2.resize(frame, (width, height)), axis=0)
    input_data = (input_data * 255).astype('uint8')  # Convert to uint8 and scale between 0-255

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output Interpretation & Handling
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype('int32')
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, ymin, xmax, ymax = int(xmin*width), int(ymin*height), int(xmax*width), int(ymax*height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            cv2.putText(frame, label, (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out.write(frame)
cap.release()
out.release()