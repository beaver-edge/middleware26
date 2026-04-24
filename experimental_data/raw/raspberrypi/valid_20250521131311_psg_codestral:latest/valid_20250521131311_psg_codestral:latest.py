import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define variables
model_path = 'models/ssd-mobilenet_v1/detect.tflite'
label_path = 'models/ssd-mobilenet_v1/labelmap.txt'
confidence_threshold = 0.5
input_path = 'data/object_detection/sheeps.mp4'
output_path = 'results/object_detection/test_results/sheeps_detections.mp4'

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Open Video File
cap = cv2.VideoCapture(input_path)
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Process Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = np.expand_dims(cv2.resize(frame, (width, height)), axis=0)

    # Set Input Tensor(s) and Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Interpret Results & Handle Output
    for i in range(len(scores)):
        if ((scores[i] > confidence_threshold) and (scores[i] <= 1.0)):
            ymin = int((boxes[i][0] * frame.shape[0]))
            xmin = int((boxes[i][1] * frame.shape[1]))
            ymax = int((boxes[i][2] * frame.shape[0]))
            xmax = int((boxes[i][3] * frame.shape[1]))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            label = '{}: {:.2f}'.format(labels[int(classes[i])], scores[i])
            cv2.putText(frame, label, (xmin, min(ymin, frame.shape[0] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Write Frame to Output Video File
    out.write(frame)

# Release Resources
cap.release()
out.release()
cv2.destroyAllWindows()