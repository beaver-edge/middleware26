import cv2
import numpy as np

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter

def load_labels(label_file):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(label_path)
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess: Resize and Normalize
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    
    # Convert to UINT8 and scale between 0 and 255 if needed
    input_data = np.expand_dims(input_data, axis=0)
    if input_dtype == np.uint8:
        input_data = np.uint8(input_data)  # Ensure the data is in UINT8 format

    # Normalize to [0, 1] if required by model (check your specific model requirements)
    input_data = input_data / 255.0
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(input_dtype))
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    height, width, _ = frame.shape
    
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            object_name = labels[int(classes[i])] if classes[i] >= 0 else 'N/A'
            cv2.putText(frame, f'{object_name}: {int(scores[i]*100)}%', 
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()