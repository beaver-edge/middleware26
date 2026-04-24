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
    print("Error opening video file")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Phase 2: Input Acquisition & Preprocessing Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (300, 300))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Phase 3: Inference
    # Assume interpreter is already loaded and allocated in setup (not shown for brevity)
    # For this example, we skip interpreter setup to focus on the input/output loop

    # Dummy inference - replace with actual interpreter invocation
    # Replace this with the actual TFLite interpreter code
    # For simplicity, let's assume the output is a list of bounding boxes
    # and class labels.
    boxes = np.array([[50, 50, 150, 150], [200, 100, 300, 200]])  # Dummy boxes
    classes = np.array([1, 2])  # Dummy classes
    scores = np.array([0.8, 0.9])  # Dummy scores

    # Phase 4: Output Interpretation & Handling Loop
    # Filter bounding boxes based on confidence threshold
    valid_boxes = []
    valid_classes = []
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            valid_boxes.append(boxes[i])
            valid_classes.append(classes[i])

    # Draw bounding boxes and labels on the frame
    for i in range(len(valid_boxes)):
        box = valid_boxes[i]
        class_id = valid_classes[i]
        label = labels[class_id]
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()