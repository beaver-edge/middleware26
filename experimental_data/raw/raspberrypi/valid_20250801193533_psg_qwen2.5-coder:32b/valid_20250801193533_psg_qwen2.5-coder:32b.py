# Phase 1: Setup
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# Phase 1.2: Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1.3: Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Phase 1.4: Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Phase 1.5: Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to match input tensor shape and dtype
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(frame_rgb, axis=0).astype(input_details[0]['dtype'])

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop

    # Get output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Assuming the model outputs a single class label index for simplicity
    predicted_index = np.argmax(output_data)
    predicted_label = labels[predicted_index]

    # Draw label on frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame with prediction to the output file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Phase 5: Cleanup
print("Processing complete. Output saved to", output_path)