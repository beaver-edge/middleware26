import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Define necessary variables using the provided model path, input path, label path, output path and confidence threshold
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels from the label file into a Python list
with open(label_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Instantiate the interpreter using the provided model path variable and allocate tensors
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Retrieve input details and output details from the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the video file for reading
cap = cv2.VideoCapture(input_path)

# Get frame width, height and fps for output video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the data to match input tensor's shape and dtype
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(input_details[0]['dtype'])

    # Set the input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor and interpret results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0].astype(np.int32)

    # Loop over the detections
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = classes[i]
            label = labels[class_id]
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width, ymin * frame_height, ymax * frame_height)
            left, right, top, bottom = map(int, [left, right, top, bottom])
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame into the file 'output.avi'
    out.write(frame)

# Release the resources acquired in phase 2 and close files and release video writer object
cap.release()
out.release()
cv2.destroyAllWindows()