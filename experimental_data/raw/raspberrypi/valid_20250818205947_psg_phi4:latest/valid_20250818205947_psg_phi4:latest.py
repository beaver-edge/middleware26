import cv2
import numpy as np

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Phase 1: Setup
# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Import TFLite Interpreter
from tflite_runtime.interpreter import Interpreter

# Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input Shape and dtype
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Output Handling: Video Writer Setup
video_capture = cv2.VideoCapture(input_path)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Phase 2: Input Acquisition & Preprocessing Loop
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Resize and Normalize the input image
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(input_dtype)
    
    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Phase 4: Output Interpretation & Handling Loop
    # Get Detection Results
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    detection_classes = interpreter.get_tensor(output_details[1]['index'])[0]
    detection_scores = interpreter.get_tensor(output_details[2]['index'])[0]

    height, width, _ = frame.shape

    for i in range(len(detection_scores)):
        if detection_scores[i] > confidence_threshold:
            # Scale box to original image dimensions
            ymin, xmin, ymax, xmax = detection_boxes[i]
            left, right, top, bottom = (xmin * width, xmax * width,
                                        ymin * height, ymax * height)
            
            class_id = int(detection_classes[i])
            label = labels[class_id]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)),
                          (0, 255, 0), 2)
            text_label = f'{label}: {detection_scores[i]:.2f}'
            cv2.putText(frame, text_label, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Handle Output: Write frame to output video
    video_writer.write(frame)

# Phase 5: Cleanup
video_capture.release()
video_writer.release()