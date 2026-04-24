import numpy as np
import cv2

# Configuration Parameters
model_path     = "models/ssd-mobilenet_v1/detect.tflite"
label_path     = "models/ssd-mobilenet_v1/labelmap.txt"
input_path     = "data/object_detection/sheeps.mp4"
output_path     = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold     = 0.5

# Load labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model and allocate tensors
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video file
cap = cv2.VideoCapture(input_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the expected input size of the model
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))

    # Preprocess the frame
    input_data = np.expand_dims(frame_resized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Post-process the output
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            label = labels[classes[i]]
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                         ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {scores[i]:.2f}', (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output file
    out.write(frame)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()