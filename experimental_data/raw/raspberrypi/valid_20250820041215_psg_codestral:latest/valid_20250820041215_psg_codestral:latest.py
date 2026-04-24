import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Setup phase
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input acquisition and preprocessing phase
cap = cv2.VideoCapture(input_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize the frame to match the expected input shape of the model
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Inference phase
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output interpretation and handling phase
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int((boxes[i][0]*height))
            xmin = int((boxes[i][1]*width))
            ymax = int((boxes[i][2]*height))
            xmax = int((boxes[i][3]*width))

            label = '{}: {:.2f}'.format(labels[int(classes[i])], scores[i])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()