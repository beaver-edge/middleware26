import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# Phase 1: Setup
model_path    = "models/ssd-mobilenet_v1/detect.tflite"
label_path    = "models/ssd-mobilenet_v1/labelmap.txt"
input_path    = "data/object_detection/sheeps.mp4"
output_path    = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold    = 0.5

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
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(input_details[0]['dtype'])

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            (ymin, xmin, ymax, xmax) = boxes[i]
            label = labels[classes[i]]
            cv2.rectangle(frame, (int(xmin * frame.shape[1]), int(ymin * frame.shape[0])),
                           (int(xmax * frame.shape[1]), int(ymax * frame.shape[0])), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()