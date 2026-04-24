import cv2
import numpy as np

# Configuration Parameters
model_path    = "models/ssd-mobilenet_v1/detect.tflite"
label_path    = "models/ssd-mobilenet_v1/labelmap.txt"
input_path    = "data/object_detection/sheeps.mp4"
output_path    = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold    = 0.5

# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter

def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to match model's expected input size without normalization
    input_size = input_details[0]['shape'][1:3]  # e.g., (300, 300)
    resized_frame = cv2.resize(frame_rgb, (input_size[1], input_size[0]))
    
    # Ensure the pixel values are in UINT8 range [0, 255]
    frame_uint8 = np.uint8(resized_frame)

    # Add batch dimension
    input_data = np.expand_dims(frame_uint8, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Phase 3: Inference
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# Note: Calculating mAP requires storing all detection results across the video and comparing them to ground truth.
# This snippet focuses on real-time processing and visualization of detections.