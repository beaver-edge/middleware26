from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# Phase 1: Setup
# 1.1 Imports: Already done above
# 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

confidence_threshold  = 0.5

# 1.3 Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 1.4 Load Interpreter
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 1.5 Get Model Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    resized_frame = cv2.resize(frame, (input_width, input_height))
    preprocessed_frame = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    # 4.1 Get Output Tensor(s)
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])
    detection_classes = interpreter.get_tensor(output_details[1]['index']).astype(np.int64)
    detection_scores = interpreter.get_tensor(output_details[2]['index'])

    # 4.2 Interpret Results
    for i, score in enumerate(detection_scores[0]):
        if score > confidence_threshold:
            box = detection_boxes[0][i]
            class_id = detection_classes[0][i]
            label = labels[class_id]

            # Draw bounding box and label on the frame
            height, width, _ = frame.shape
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4.3 Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()