import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define paths and parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load TensorFlow Lite model
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Open video file
cap = cv2.VideoCapture(input_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = np.array(resized_frame, dtype=np.float32) / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0).astype(input_dtype)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret results (not mentioned in the prompt)
    # Handle output (display or save to file)

    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()