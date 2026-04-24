import cv2
from ai_edge_litert.interpreter import Interpreter
import numpy as np

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape and dtype
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Open video file
video_capture = cv2.VideoCapture(input_path)

# Get frame width, height, and fps for output video writer
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = int(video_capture.get(5))
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Phase 2: Input Acquisition & Preprocessing Loop
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize and normalize frame
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = np.array(resized_frame, dtype=np.float32) / 255.0

    # Convert frame to match input tensor shape and dtype
    input_data = np.expand_dims(normalized_frame, axis=0).astype(input_dtype)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling
    output_boxes = interpreter.get_tensor(output_details[0]['index'])
    output_classes = interpreter.get_tensor(output_details[1]['index'])
    output_scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    for i in range(num_detections):
        score = output_scores[0, i]
        if score >= confidence_threshold:
            class_id = int(output_classes[0, i])
            label = labels[class_id]
            box = output_boxes[0, i] * [frame_height, frame_width, frame_height, frame_width]
            y1, x1, y2, x2 = box.astype(np.int32)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    output_video.write(frame)

# Phase 5: Cleanup
video_capture.release()
output_video.release()
cv2.destroyAllWindows()