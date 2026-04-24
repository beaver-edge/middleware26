import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define necessary variables using the provided paths and parameters
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load labels from the label file into a Python list
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f]

# Instantiate and allocate tensors using the provided model path variable
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Retrieve input and output details from the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video capture and video writer for processing and outputting frames
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Process video frames continuously until there are no more left
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data to match the shape and dtype specified in the input details
    resized_frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8) # changed the dtype to uint8 and removed normalization

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve output tensor, interpret results, and handle output
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index']).astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])

    for i, score in enumerate(scores[0]):
        if score > confidence_threshold:
            class_id = classes[0][i]
            label = labels[class_id]
            ymin, xmin, ymax, xmax = boxes[0][i]
            left = int(xmin * frame.shape[1])
            top = int(ymin * frame.shape[0])
            right = int(xmax * frame.shape[1])
            bottom = int(ymax * frame.shape[0])
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out.write(frame)

# Release resources acquired for input and output
cap.release()
out.release()
cv2.destroyAllWindows()