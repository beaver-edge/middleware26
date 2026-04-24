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

# 2.1 Acquire Input Data (Video Processing Setup)
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Phase 3: Inference

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame to match input tensor shape and dtype
    input_frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_frame = np.expand_dims(input_frame, axis=0)
    input_frame = input_frame.astype(input_details[0]['dtype'])

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_frame)

    # Run inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # 4.1 Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 4.2 Interpret Results
    # Assuming the model outputs a list of class probabilities for each frame
    # For demonstration, let's just find the class with the highest probability
    scores = np.max(output_data, axis=1)
    predicted_class_index = np.argmax(scores)
    predicted_label = labels[predicted_class_index]

    # 4.3 Handle Output
    # Add the label to the frame and write it to the output video file
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)

# Phase 5: Cleanup

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved to:", output_path)