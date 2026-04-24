import cv2
from ai_edge_litert.interpreter import Interpreter
import numpy as np

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_map_path  = "models/ssd-mobilenet_v1/labelmap.txt"

with open(label_map_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = "data/object_detection/sheeps.mp4"
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture('path_to_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data (resize and convert to numpy array)
    input_data = np.expand_dims(cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2])), axis=0).astype(np.float32)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = labels[np.argmax(output_data)]

    # Handle output (display or save results)
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Phase 5: Cleanup
cap.release()
cv2.destroyAllWindows()