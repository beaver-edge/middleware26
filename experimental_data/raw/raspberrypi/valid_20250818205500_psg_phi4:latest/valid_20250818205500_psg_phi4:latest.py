import cv2
import numpy as np
import tensorflow as tf

# Phase 1: Setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# Load labels
with open(label_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

# Create a video writer to save the output with detections
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the input frame to match the model's expected size
    resized_frame = cv2.resize(frame, (300, 300))
    
    # Convert image to UINT8 and ensure it matches the required input format for the model
    # Assuming the model expects inputs as UINT8 without normalization
    image_np_expanded = np.expand_dims(resized_frame.astype(np.uint8), axis=0)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], image_np_expanded)
    interpreter.invoke()

    # Extract output data
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Phase 4: Output Interpretation & Handling Loop
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            class_id = int(classes[i])
            box = boxes[i]
            ymin, xmin, ymax, xmax = box

            # Scale the bounding box back to the original frame size
            left, right, top, bottom = (xmin * frame_width, xmax * frame_width,
                                        ymin * frame_height, ymax * frame_height)

            label = labels[class_id]
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()