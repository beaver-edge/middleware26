import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Define paths/parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
video_path = 'path_to_input_video.mp4'
label_map_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load labels
with open(label_map_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details  = "data/object_detection/sheeps.mp4"
output_details = interpreter.get_output_details()

# Open video file and output text file
cap = cv2.VideoCapture(video_path)
with open(output_path, 'w') as f:
    # Input acquisition & preprocessing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess data
        input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Output interpretation & handling loop
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=1)[0]
        label = labels[predicted_class]
        f.write(f'{label}\n')

# Cleanup
cap.release()