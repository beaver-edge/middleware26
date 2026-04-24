import cv2
import numpy as np

# Phase 1: Setup
# 1.2. Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold  = 0.5

# 1.3. Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = None
try:
    from tfLite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except ImportError:
    print("TFLite runtime not found. Please install it.")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Phase 4: Output Interpretation & Handling Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess the frame
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0)

    # Convert to float32 and normalize
    input_data = input_data.astype(np.float32)
    input_data = input_data / 255.0  # Normalize to [0, 1]

    # Check the expected type
    expected_type = input_details[0]['dtype']
    if expected_type == np.uint8:
        input_data = input_data.astype(np.uint8)  # Convert to uint8 if needed

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process output and draw bounding boxes
    boxes = output_data[0][0:4]
    classes = output_data[0][4]
    scores = output_data[0][5]

    # The error was caused by the fact that scores is a numpy array. To check if the condition is satisfied we should use scores.any() instead of scores > confidence_threshold
    if scores.any() > confidence_threshold:
        class_id = int(classes)
        if 0 <= class_id < len(labels):
            label = labels[class_id]
            x1, y1, x2, y2 = boxes
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {scores:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()