import cv2
import numpy as np
import tensorflow as tf

# Configuration Parameters
model_path    = "models/ssd-mobilenet_v1/detect.tflite"
label_path    = "models/ssd-mobilenet_v1/labelmap.txt"
input_path    = "data/object_detection/sheeps.mp4"
output_path    = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold    = 0.5

# Load Labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Video capture and writer setup
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Initialize variables for mAP calculation
detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_shape = input_details[0]['shape']
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (input_shape[1], input_shape[2]))
    
    # Ensure the image is in UINT8 format and scale pixel values to [0, 255]
    # Note: This step ensures compatibility with the model's expected input type
    if image_rgb.dtype != np.uint8:
        resized_image = cv2.convertScaleAbs(resized_image)

    input_data = np.expand_dims(resized_image, axis=0).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence

    # Draw rectangles and labels on detected objects
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)

            class_id = int(classes[i])
            label = f'{labels[class_id - 1]}: {int(scores[i] * 100)}%'
            cv2.rectangle(frame, (left, top), (right, bottom), (23, 230, 210), thickness=2)
            cv2.putText(frame, label, (left + 10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Collect detections for mAP calculation
    detections.append((classes, scores, boxes))

    # Write the frame with annotations to output video
    out.write(frame)

cap.release()
out.release()

# Function to calculate mAP (placeholder for actual implementation)
def calculate_mAP(detections):
    # Placeholder: Implement mAP calculation logic here
    return 0.75  # Example value

# Calculate and print mAP
mAP = calculate_mAP(detections)
print(f'mAP: {mAP}')