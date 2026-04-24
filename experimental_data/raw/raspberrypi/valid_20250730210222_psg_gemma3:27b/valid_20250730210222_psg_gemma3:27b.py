from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# Phase 1: Setup
# 1.2. Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# 1.3. Load Labels (Conditional)
try:
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]
except FileNotFoundError:
    labels = None  # Labels are not available

# 1.4. Load Interpreter
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()

# 1.5. Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data_type = input_details[0]['dtype']

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1. Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Phase 4: Output Interpretation & Handling Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2.2. Preprocess Data
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0)

    if input_data_type == np.float32:
        input_data = input_data.astype(np.float32) / 255.0  # Normalize if needed

    # 3.1. Set Input Tensor(s)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 3.2. Run Inference
    interpreter.invoke()

    # 4.1. Get Output Tensor(s)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 4.2. Interpret Results
    # Assuming output_data contains bounding boxes and class IDs
    # Process the output data to extract bounding boxes and class IDs

    # Placeholder for processing output data
    # Replace with your specific processing logic
    # Example:
    # boxes = output_data[..., 0:4]
    # classes = output_data[..., 4]

    # 4.3. Handle Output
    # Draw bounding boxes and labels on the frame
    # Example:
    # for i in range(boxes.shape[0]):
    #     box = boxes[i]
    #     class_id = int(classes[i])
    #     label = labels[class_id] if labels else str(class_id)
    #     cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    #     cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()