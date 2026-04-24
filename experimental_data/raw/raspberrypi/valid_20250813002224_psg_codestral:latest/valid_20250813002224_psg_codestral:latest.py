import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define paths/parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter and get model details
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Open video stream
cap = cv2.VideoCapture(input_path)
while cap.isOpened():
    # Acquire input data and preprocess it
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Interpret results and handle output
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    for i, score in enumerate(scores):
        if score > confidence_threshold:
            class_id = int(classes[i])
            label = labels[class_id]
            ymin, xmin, ymax, xmax = boxes[i]
            # Draw bounding box and label on frame
            cv2.rectangle(frame, (int(xmin * frame.shape[1]), int(ymin * frame.shape[0])),
                          (int(xmax * frame.shape[1]), int(ymax * frame.shape[0])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Save frame with annotated objects to output folder
    cv2.imwrite(f"{output_path}/frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg", frame)