import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Paths/Parameters
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load Labels
with open(label_path, 'r') as f:
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
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    input_data = cv2.resize(input_data, (width, height))  # Resize
    input_data = np.expand_dims(input_data, axis=0)  # Expand dimensions
    input_data = (input_data / 255).astype('uint8')  # Normalize and scale to [0, 255]

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i, score in enumerate(scores):
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            label = labels[int(classes[i])]
            cv2.rectangle(frame, (int(xmin * frame_size[0]), int(ymin * frame_size[1])), (int(xmax * frame_size[0]), int(ymax * frame_size[1])), (10, 255, 0), 4)
            cv2.putText(frame, label, (int(xmin * frame_size[0]), int(ymin * frame_size[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()