import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Configuration Parameters
model_path   = "models/ssd-mobilenet_v1/detect.tflite"
label_path   = "models/ssd-mobilenet_v1/labelmap.txt"
input_path   = "data/object_detection/sheeps.mp4"
output_path   = "results/object_detection/test_results/sheeps_detections.mp4"
confidence_threshold   = 0.5

# Phase 1: Setup
def load_labels(label_file):
    with open(label_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

labels = load_labels(label_path)

interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Phase 2: Input Acquisition & Preprocessing Loop
def preprocess_frame(frame):
    input_shape = input_details[0]['shape']
    
    # Resize frame to match the model's expected input size
    image = cv2.resize(frame, (input_shape[1], input_shape[2]))
    
    # Normalize the pixel values to [0, 1] and convert to UINT8
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions to match model's expected input shape
    image = np.expand_dims(image, axis=0).astype(np.uint8)
    
    return image

# Phase 3: Inference
def run_inference(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    return boxes, classes, scores

# Phase 4: Output Interpretation & Handling Loop
def draw_detections(frame, boxes, classes, scores):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            object_name = labels[int(classes[i])]
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (10, 255, 0), 2)
            label = f'{object_name}: {int(scores[i] * 100)}%'
            cv2.putText(frame, label, (int(left), int(top) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 0), 2)

# Phase 5: Cleanup
def main():
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_data = preprocess_frame(frame)
        boxes, classes, scores = run_inference(input_data)
        draw_detections(frame, boxes, classes, scores)

        out.write(frame)
        
        # Comment or remove the next line to avoid GUI display error
        # cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

# Execute the main function
if __name__ == "__main__":
    main()