import numpy as np
from ai_edge_litert.interpreter import Interpreter
import cv2

# Define necessary variables using the provided model path and other parameters
model_path      = "models/ssd-mobilenet_v1/detect.tflite"
label_path      = "models/ssd-mobilenet_v1/labelmap.txt"
input_path      = "data/object_detection/sheeps.mp4"
output_path     = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold      = 0.5

# Load Labels
labels = []
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video capture and output file for writing
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Output Tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Interpret Results and Handle Output
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin = int((boxes[i][0] * frame.shape[0]))
            xmin = int((boxes[i][1] * frame.shape[1]))
            ymax = int((boxes[i][2] * frame.shape[0]))
            xmax = int((boxes[i][3] * frame.shape[1]))

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            cv2.putText(frame, label, (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)

    # Write frame to output file
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()