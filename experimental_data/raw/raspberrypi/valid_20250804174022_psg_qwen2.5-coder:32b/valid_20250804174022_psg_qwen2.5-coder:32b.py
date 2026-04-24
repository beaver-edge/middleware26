# Phase 1: Setup

# Imports
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop

# Acquire Input Data
cap = cv2.VideoCapture(input_path)

# Video writer setup for output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess Data
    input_shape = input_details[0]['shape']
    preprocessed_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    preprocessed_frame = preprocessed_frame.astype(input_details[0]['dtype'])
    
    if len(input_shape) == 4:
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    # Phase 3: Inference
    
    # Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    
    # Run Inference
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    
    # Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Interpret Results (Assuming the model outputs class probabilities or logits)
    scores = np.squeeze(output_data)
    predicted_class_index = np.argmax(scores)
    predicted_label = labels[predicted_class_index]
    
    # Handle Output
    cv2.putText(frame, f'Class: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(frame)

# Phase 5: Cleanup

cap.release()
out.release()
cv2.destroyAllWindows()