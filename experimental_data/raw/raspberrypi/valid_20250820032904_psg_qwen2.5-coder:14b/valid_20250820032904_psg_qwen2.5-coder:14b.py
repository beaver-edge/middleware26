import numpy as np
from tflite_runtime.interpreter import Interpreter
import cv2

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    image_np = np.array(frame)
    input_data = cv2.resize(image_np, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = (input_data - 127.5) / 127.5  # Normalize to [-1, 1]
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(input_details[0]['dtype'])

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = int(classes[i])
            label = labels[class_id]
            box = boxes[i]
            y_min, x_min, y_max, x_max = map(int, box * [image_np.shape[1], image_np.shape[0], image_np.shape[1], image_np.shape[0]])
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_np, f'{label}: {scores[i]:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(image_np)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()