import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Define necessary variables
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
labelmap_path = 'models/ssd-mobilenet_v1/labelmap.txt'
confidence_threshold = 0.5
input_video_path = 'data/object_detection/sheeps.mp4'
output_video_path = 'results/object_detection/test_results/sheeps_detections.mp4'

# Load Labels
with open(labelmap_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Instantiate Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]

# Acquire Input Data
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = np.expand_dims(cv2.resize(frame, (width, height)), axis=0).astype(np.uint8)

    # Set Input Tensor & Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Output Tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Interpret Results & Handle Output
    for i, score in enumerate(scores):
        if score > confidence_threshold:
            cv2.rectangle(frame, (int(boxes[i][1]*width), int(boxes[i][0]*height)),
                          (int(boxes[i][3]*width), int(boxes[i][2]*height)), (0, 255, 0), 2)
            cv2.putText(frame, labels[int(classes[i])],
                        (int(boxes[i][1]*width), int(boxes[i][0]*height)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()