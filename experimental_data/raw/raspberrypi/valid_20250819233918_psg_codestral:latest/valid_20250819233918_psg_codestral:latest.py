import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Load model and labels
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = "data/object_detection/sheeps.mp4"
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"
labels = open(label_path).read().splitlines() # assuming labels are in a text file with one label per line

# Open video file and create output file
video_path = 'your_video_path.mp4'  # replace with your video path
cap = cv2.VideoCapture(video_path)
output_path = 'output_video_path.mp4'   # replace with your desired output video path
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image for input to our model
    input_data = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5 # assuming model expects inputs in range [-1, 1]

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Interpret results and draw rectangles on the detected objects
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.squeeze(output_data)
    label_id = np.argmax(predictions)
    confidence = predictions[label_id]
    if confidence > 0.5: # assuming we only care about detections with high confidence
        label = labels[label_id]
        cv2.putText(frame, f'{label} {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # assuming model returns bounding boxes in the format [y1, x1, y2, x2] and input shape is [480, 640]
        h, w = frame.shape[:2]
        bbox = output_data[0]['locations'][0].numpy() * np.array([h, w, h, w]) # convert to pixels
        y1, x1, y2, x2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write frame to output video file
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()