import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Define paths and parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
video_path = 'path_to_input_video.mp4'
label_map_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Load labels
with open(label_map_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter and get model details
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = "data/object_detection/sheeps.mp4"
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

# Open video file and loop through each frame
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_data = np.expand_dims(frame, axis=0).astype(input_details[0]['dtype'])

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output and interpret results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = labels[np.argmax(output_data)]

    # Handle output (e.g., display on screen)
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()