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

# 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)

# Prepare video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 2.2 Preprocess Data
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Assuming the model expects input of shape [1, H, W, C]
    preprocessed_frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
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
    # Assuming the model outputs class probabilities in a single array for each input frame
    predictions = np.squeeze(output_data)  # Remove extra dimensions
    
    # Find the index of the highest probability and get the corresponding label
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    
    # Draw the predicted label on the frame
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 4.3 Handle Output
    out.write(frame)

# Phase 5: Cleanup

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()