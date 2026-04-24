import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

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

# Input Tensor Info
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Output Tensor Info
boxes_idx, classes_idx, scores_idx = 0, 1, 2
boxes_tensor = output_details[boxes_idx]['index']
classes_tensor = output_details[classes_idx]['index']
scores_tensor = output_details[scores_idx]['index']

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(cv2.resize(frame_rgb, (input_shape[1], input_shape[2])), axis=0).astype(input_dtype)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(boxes_tensor)[0]
    classes = interpreter.get_tensor(classes_tensor)[0]  # Removed label_offset since it's not defined
    scores = interpreter.get_tensor(scores_tensor)[0]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            box = boxes[i]
            class_id = int(classes[i])
            label = labels[class_id]

            # Draw bounding box and label
            ymin, xmin, ymax, xmax = map(int, box * [frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()