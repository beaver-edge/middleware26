import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# Phase 1: Setup

# Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Phase 2: Input Acquisition & Preprocessing Loop

# Acquire Input Data
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = input_data[:, :, ::-1]  # Convert BGR to RGB
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data / 127.5) - 1.0  # Normalize to [-1, 1]
    
    if input_dtype == np.uint8:
        input_data = input_data.astype(np.uint8)
    elif input_dtype == np.float32:
        input_data = input_data.astype(np.float32)

    # Phase 3: Inference

    # Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Interpret Results
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            label = labels[classes[i]]
            ymin, xmin, ymax, xmax = boxes[i]
            (startX, startY) = (int(xmin * frame_width), int(ymin * frame_height))
            (endX, endY) = (int(xmax * frame_width), int(ymax * frame_height))

            # Draw the bounding box and label on the image
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup

cap.release()
out.release()
cv2.destroyAllWindows()