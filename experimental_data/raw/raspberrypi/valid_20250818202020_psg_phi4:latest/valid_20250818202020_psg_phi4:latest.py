import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Setup Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
floating_model = input_details[0]['dtype'] == np.float32

# Helper function to preprocess the frame
def preprocess_frame(frame, input_shape):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_shape[1], input_shape[2]))
    
    if floating_model:
        frame_normalized = np.float32(frame_resized) / 255.0
    else:
        frame_normalized = np.uint8(frame_resized)

    return np.expand_dims(frame_normalized, axis=0)

# Open Video Capture and Prepare Output Writer
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

detections_list = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the input
    input_data = preprocess_frame(frame, input_shape)
    
    # Set model input and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (int(left), int(top - label_size[1])), (int(left + label_size[0]), int(top + base_line)), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Collecting detection details for performance evaluation
            detections_list.append((object_name, scores[i]))

    out.write(frame)

# Release resources
cap.release()
out.release()

# Calculate mean Average Precision (mAP) as a performance metric example
from collections import defaultdict

def calculate_map(detections_list, iou_threshold=0.5):
    # Placeholder function for mAP calculation
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    total_detections = defaultdict(int)

    # For simplicity assuming all detections are correct and counting them as true positives
    for detection in detections_list:
        object_name, _ = detection
        true_positive[object_name] += 1

    # Assuming a fixed number of ground truths per video frame for evaluation
    num_ground_truths_per_frame = 5
    total_frames = len(detections_list) // num_ground_truths_per_frame

    for label in labels:
        total_detections[label] = num_ground_truths_per_frame * total_frames

    # Calculating precision and recall (basic example)
    precision = {label: true_positive[label] / max(true_positive[label] + false_positive[label], 1) 
                 for label in labels}
    recall = {label: true_positive[label] / max(total_detections[label], 1) 
              for label in labels}

    # Calculate Average Precision (AP) for each class
    ap_per_class = {}
    for label in labels:
        if total_detections[label] == 0:
            continue
        ap_per_class[label] = (precision.get(label, 0) + recall.get(label, 0)) / 2

    # Calculate mean AP
    map_score = np.mean(list(ap_per_class.values()))
    
    return map_score

mean_ap = calculate_map(detections_list)
print(f'Mean Average Precision (mAP): {mean_ap:.2f}')