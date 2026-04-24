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

# Load TFLite model
try:
    interpreter = cv2.dnn.readNetFromTflite(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 1.5 Get Model Details
input_size = interpreter.getNetworkInputs()[0].shape[:2]

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Data
    resized_frame = cv2.resize(frame, (input_size[0], input_size[1]))
    input_data = np.expand_dims(resized_frame, axis=0)
    input_data = np.array(input_data, dtype=np.float32)

    # Phase 3: Inference
    interpreter.setTensor(0, input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.getTensor(1)
    classes = interpreter.getTensor(2)
    scores = interpreter.getTensor(3)

    for i in range(boxes.shape[0]):
        if scores[i][0] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i][0]
            class_id = int(classes[i][0])
            label = labels[class_id]
            
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()