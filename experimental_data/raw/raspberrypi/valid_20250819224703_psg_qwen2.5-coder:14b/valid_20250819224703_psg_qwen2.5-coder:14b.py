import numpy as np
from tflite_runtime.interpreter import Interpreter
import cv2

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
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
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    image_np = cv2.resize(frame, (input_shape[1], input_shape[2]))
    image_np_expanded = np.expand_dims(image_np, axis=0)
    input_data = image_np_expanded.astype(input_dtype)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for box, cls_id, score in zip(boxes, classes, scores):
        if score > confidence_threshold:
            y_min, x_min, y_max, x_max = box
            label = labels[int(cls_id)]
            cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])),
                          (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()