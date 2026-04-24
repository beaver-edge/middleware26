import numpy as np
import cv2
import tensorflow as tf

# Phase 1: Setup

# Paths/Parameters
model_path     = "models/ssd-mobilenet_v1/detect.tflite"
label_path     = "models/ssd-mobilenet_v1/labelmap.txt"
input_path     = "data/object_detection/sheeps.mp4"
output_path     = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold     = 0.5

# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load Interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop

# Open video file
cap = cv2.VideoCapture(input_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data - 127.5) / 127.5
    input_data = input_data.astype(np.uint8)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # Get Output Tensor(s)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Interpret Results
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            y_min, x_min, y_max, x_max = boxes[i]
            label = labels[int(classes[i])]
            score = scores[i]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])),
                          (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup

cap.release()
out.release()
cv2.destroyAllWindows()