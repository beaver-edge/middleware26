import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

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

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    height, width = frame.shape[:2]
    input_data = cv2.resize(frame, (300, 300))
    input_data = input_data.astype('float32')
    input_data /= 127.5
    input_data -= 1.0
    input_data = np.expand_dims(input_data, axis=0)

    # Ensure the input data type matches the expected type by the model
    if input_details[0]['dtype'] == np.uint8:
        input_data = (input_data * 127.5 + 127.5).astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Phase 3: Inference
    interpreter.invoke()

    # Get outputs from the model
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Removed label_offset as it was not defined
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Phase 4: Output Interpretation & Handling Loop
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            (left, top, right, bottom) = boxes[i]
            left *= width
            top *= height
            right *= width
            bottom *= height

            label_id = int(classes[i])
            label_name = labels[label_id]

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label_name}: {scores[i]:.2f}', (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()