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
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get model input size
input_shape = input_details[0]['shape']
input_size = (input_shape[1], input_shape[2])

def process_frame(frame):
    # Resize frame to match model input size
    blob = cv2.resize(frame, input_size)

    # Normalize pixel values from [0, 255] to [0, 1]
    blob = blob.astype(np.float32) / 255.0

    # Expand dimensions and ensure it is of type UINT8
    blob = np.expand_dims(blob, axis=0).astype(np.uint8)

    # Set the tensor for the frame
    interpreter.set_tensor(input_details[0]['index'], blob)

    # Run inference
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    confidences = interpreter.get_tensor(output_details[1]['index'])[0]  # Confidence scores for detected objects
    class_ids = interpreter.get_tensor(output_details[2]['index'])[0]  # Class index for detected objects

    # Filter detections based on confidence threshold
    filtered_boxes = []
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        if confidence > confidence_threshold:
            label = labels[int(class_id)]
            x1, y1, x2, y2 = int(box[1] * frame.shape[1]), int(box[0] * frame.shape[0]), \
                             int(box[3] * frame.shape[1]), int(box[2] * frame.shape[0])
            filtered_boxes.append((label, (x1, y1, x2, y2), confidence))

    return filtered_boxes

# Open video file
video_capture = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (int(video_capture.get(3)), int(video_capture.get(4))))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process the frame
    detections = process_frame(frame)

    # Draw bounding boxes and labels on the frame
    for label, box, confidence in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'{label}: {confidence:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    output_video.write(frame)

# Release resources
video_capture.release()
output_video.release()
cv2.destroyAllWindows()