# Phase 1: Setup

# 1.1 Imports
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# 1.3 Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop

# Initialize video capture
cap = cv2.VideoCapture(input_path)

# Define codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2.2 Preprocess Data
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    preprocessed_frame = np.expand_dims(resized_frame, axis=0)
    preprocessed_frame = preprocessed_frame.astype(input_details[0]['dtype'])

    # Phase 3: Inference

    # 3.1 Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

    # 3.2 Run Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # 4.1 Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 4.2 Interpret Results
    detections = np.squeeze(output_data)

    for detection in detections:
        if detection[2] > 0.5:  # Confidence threshold
            label_id = int(detection[1])
            label_name = labels[label_id]
            score = detection[2]
            bbox = detection[3:]

            # Calculate bounding box coordinates
            y_min, x_min, y_max, x_max = bbox * [height, width, height, width]

            # 4.3 Handle Output
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label_text = f'{label_name}: {score:.2f}'
            cv2.putText(frame, label_text, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    out.write(frame)

# Phase 5: Cleanup

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()