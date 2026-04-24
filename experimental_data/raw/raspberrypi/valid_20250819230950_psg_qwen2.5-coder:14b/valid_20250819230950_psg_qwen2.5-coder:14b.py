import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# Phase 1: Setup

# 1.1. Imports
# No additional imports needed for this task.

# 1.2. Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3. Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4. Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5. Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Phase 2: Input Acquisition & Preprocessing Loop

# 2.1. Acquire Input Data
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise IOError("Cannot open video")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 2.2. Preprocess Data
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = resized_frame / 127.5 - 1.0
    return np.expand_dims(normalized_frame, axis=0).astype(input_dtype)

# Phase 3: Inference

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    input_data = preprocess_frame(frame)
    
    # 3.1. Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 3.2. Run Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # 4.1. Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # 4.2. Interpret Results
    for i in range(len(scores[0])):
        if scores[0][i] > confidence_threshold:
            class_id = int(classes[0][i])
            box = boxes[0][i]
            ymin, xmin, ymax, xmax = map(int, box * [frame_height, frame_width, frame_height, frame_width])

            label = f"{labels[class_id]}: {scores[0][i]:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 4.3. Handle Output
    out.write(frame)

# Phase 5: Cleanup

cap.release()
out.release()
cv2.destroyAllWindows()