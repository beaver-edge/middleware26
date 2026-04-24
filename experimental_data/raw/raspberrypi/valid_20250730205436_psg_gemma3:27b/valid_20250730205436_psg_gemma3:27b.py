import cv2
import numpy as np

# Phase 1: Setup
# 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3 Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize video capture and output
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Placeholder for interpreter initialization (not needed for this specific task)
# interpreter = Interpreter(model_path=model_path)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing (resize and normalize)
    img = cv2.resize(frame, (300, 300))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Phase 3: Inference (Placeholder - No actual inference in this example)
    # interpreter.set_tensor(input_details[0]['index'], img)
    # interpreter.invoke()
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Placeholder for dummy detection
    detections = []
    if np.random.rand() > 0.5:
        detections.append((np.random.randint(0, frame_width), np.random.randint(0, frame_height), np.random.rand(), labels[np.random.randint(0, len(labels))]))

    # Phase 4: Output Interpretation & Handling Loop
    for (x, y, score, label) in detections:
        if score > confidence_threshold:
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the output frame
    out.write(frame)
    
# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()