import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# Phase 1: Setup
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

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = input_data.reshape((1, input_details[0]['shape'][1], input_details[0]['shape'][2], 3)).astype(np.uint8)

    # Normalize if required by the model
    if 'mean' in input_details[0] and 'std' in input_details[0]:
        mean = np.array(input_details[0]['mean']).reshape((1, 1, 3))
        std = np.array(input_details[0]['std']).reshape((1, 1, 3))
        input_data = (input_data - mean) / std

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    num_detections = interpreter.get_tensor(output_details[3]['index'])[0]  # Number of detections

    # Interpret Results
    for i in range(int(num_detections)):
        if scores[i] > confidence_threshold:
            class_name = labels[int(classes[i])]
            y_min, x_min, y_max, x_max = boxes[i]
            cv2.rectangle(frame, (int(x_min * width), int(y_min * height)), (int(x_max * width), int(y_max * height)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {scores[i]:.2f}', (int(x_min * width), int(y_min * height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()