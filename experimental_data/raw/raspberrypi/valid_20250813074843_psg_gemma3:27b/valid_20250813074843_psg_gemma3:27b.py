import cv2
import numpy as np

# Phase 1: Setup
# 1.2. Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3. Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize video capture and output
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Load the TFLite model
try:
    interpreter = cv2.dnn.readNetFromTflite(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Phase 2: Input Acquisition & Preprocessing Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (300, 300))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)

    # Phase 3: Inference
    interpreter.setInput(img)
    output = interpreter.forward()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = output[0, 0, :, 3:7]
    classes = output[0, 0, :, 5]
    scores = output[0, 0, :, 4]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
            class_id = int(classes[i])
            label = labels[class_id]
            confidence = scores[i]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()