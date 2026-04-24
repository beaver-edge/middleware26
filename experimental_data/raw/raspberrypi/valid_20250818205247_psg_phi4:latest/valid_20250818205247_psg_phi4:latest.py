import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Configuration Parameters
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Phase 1: Setup
class Interpreter:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def get_input_details(self):
        return self.interpreter.get_input_details()

    def get_output_details(self):
        return self.interpreter.get_output_details()

    def set_tensor(self, index, value):
        tensor = self.interpreter.tensor(index)()[0]
        np.copyto(tensor, value)

    def invoke(self):
        self.interpreter.invoke()
    
    def get_tensor(self, index):
        return self.interpreter.get_tensor(index)

# Load labels
def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)
interpreter = Interpreter(model_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check the input tensor's expected data type
expected_dtype = input_details[0]['dtype']

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input
    input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(input_data, axis=0)
    
    # Normalize to [0, 1] if required by the model
    input_data = input_data / 255.0

    # Convert data type based on expected dtype
    if expected_dtype == np.float32:
        input_data = input_data.astype(np.float32)
    elif expected_dtype == np.uint8:
        input_data = (input_data * 255).astype(np.uint8)

    # Phase 3: Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Phase 4: Output Interpretation & Handling Loop
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{labels[int(classes[i])]}: {int(scores[i]*100)}%'
            cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Handle Output
    out.write(frame)
    
    # Removed cv2.imshow('Object Detection', frame) and cv2.waitKey(1) to avoid display issues

# Phase 5: Cleanup
cap.release()
out.release()