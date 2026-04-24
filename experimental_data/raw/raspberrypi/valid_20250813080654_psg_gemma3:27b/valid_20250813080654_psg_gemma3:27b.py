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

# Load TFLite model and allocate tensors.
interpreter = None
try:
  from tfLite_runtime.interpreter import Interpreter
  interpreter = Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
except ImportError:
  print("Please install tfLite_runtime: pip install tfLite_runtime")
  exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
  print("Error opening video file")
  exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Phase 4: Output Interpretation & Handling Loop
while True:
  ret, frame = cap.read()
  if not ret:
    break

  # Resize and preprocess frame
  resized_frame = cv2.resize(frame, (300, 300))
  input_data = np.expand_dims(resized_frame, axis=0)
  # Correct the data type. The error message indicates the expected type is UINT8
  input_data = input_data.astype(np.uint8)
  # Normalization is usually required, but the error is about the data type. Leaving this out for now to fix the primary error.
  # input_data /= 255.0

  # Set input tensor
  interpreter.set_tensor(input_details[0]['index'], input_data)

  # Run inference
  interpreter.invoke()

  # Get output tensor
  output_data = interpreter.get_tensor(output_details[0]['index'])

  # Process output data
  boxes = output_data[0][0][:, 1:5]
  classes = output_data[0][0][:, 5]
  scores = output_data[0][0][:, 6]

  for i in range(len(scores)):
    if scores[i] > confidence_threshold:
      ymin, xmin, ymax, xmax = boxes[i]
      xmin = int(xmin * frame.shape[1])
      xmax = int(xmax * frame.shape[1])
      ymin = int(ymin * frame.shape[0])
      ymax = int(ymax * frame.shape[0])

      label = labels[int(classes[i])]
      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
      cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()