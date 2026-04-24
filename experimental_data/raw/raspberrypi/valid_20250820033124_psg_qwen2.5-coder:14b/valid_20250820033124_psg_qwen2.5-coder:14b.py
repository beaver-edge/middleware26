import numpy as np
import cv2
import tensorflow as tf

# Phase 1: Setup
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess data
    input_height, input_width = input_details[0]['shape'][1:3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_width, input_height))
    input_data = np.expand_dims(frame_resized, axis=0).astype(input_details[0]['dtype'])

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Assuming the first output is bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)  # Assuming the second output is class indices
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Assuming the third output is scores

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            y_min, x_min, y_max, x_max = boxes[i]
            label = labels[classes[i]]
            cv2.rectangle(frame, (int(x_min * input_width), int(y_min * input_height)), 
                             (int(x_max * input_width), int(y_max * input_height)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', 
                         (int(x_min * input_width), int(y_min * input_height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                         0.9, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()