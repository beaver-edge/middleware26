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
   net = cv2.dnn.readNetFromTflite(model_path)
except Exception as e:
   print(f"Error loading model: {e}")
   exit()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
   print("Error opening video file")
   exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

while True:
   ret, frame = cap.read()
   if not ret:
       break

   # Preprocess the frame
   blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

   # Phase 3: Inference
   net.setInput(blob)
   detections = net.forward()

   # Phase 4: Output Interpretation & Handling
   for detection in detections[0]:
       confidence = detection[5]
       if confidence > confidence_threshold:
           class_id = int(detection[0])
           box = detection[1:5] * np.array([frame_width, frame_height, frame_width, frame_height])
           (xmin, ymin, xmax, ymax) = box.astype(int)
           label = labels[class_id]
           cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
           cv2.putText(frame, f"{label}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()