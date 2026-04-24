import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define necessary variables using the provided parameters
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load Labels
labels = []
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open Video File and output file for writing
cap = cv2.VideoCapture(input_path)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5 # Normalization here
    input_data = np.uint8(input_data * 255) # Convert back to UINT8

    # Set Input Tensor and Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Output Tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    # Interpret Results and Handle Output
    for i in range(num_detections):
        if scores[0, i] > confidence_threshold:
            label = labels[int(classes[0, i])]
            box = boxes[0, i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            start_point = (int(box[1]), int(box[0]))
            end_point = (int(box[3]), int(box[2]))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(frame, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    out.write(frame)
cap.release()
out.release()