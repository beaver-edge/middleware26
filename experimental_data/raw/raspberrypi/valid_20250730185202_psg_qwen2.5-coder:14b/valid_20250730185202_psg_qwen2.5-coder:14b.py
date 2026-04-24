import numpy as np
from ai_edge_litert.interpreter import Interpreter

# **Phase 1: Setup**
# 1.1. Imports: Already imported necessary libraries above.
# 1.2. Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# 1.3. Load Labels (Conditional)
labels = []
if label_path:
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

# 1.4. Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5. Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# **Phase 2: Input Acquisition & Preprocessing Loop**
# Assuming input is a video file, we will process each frame.
import cv2

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to match input tensor requirements
    # Example: Resize and normalize
    height, width, channels = input_details[0]['shape'][1], input_details[0]['shape'][2], input_details[0]['shape'][3]
    resized_frame = cv2.resize(frame, (width, height))
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    input_data = np.expand_dims(normalized_frame, axis=0).astype(input_details[0]['dtype'])

    # **Phase 3: Inference**
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # **Phase 4: Output Interpretation & Handling Loop**
    # 4.1. Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 4.2. Interpret Results
    # Assuming output is a classification result, get the index of the highest score
    predicted_index = np.argmax(output_data)
    predicted_label = labels[predicted_index] if labels else f"Class {predicted_index}"

    # 4.3. Handle Output
    print(f"Predicted Label: {predicted_label}")

# **Phase 5: Cleanup**
cap.release()