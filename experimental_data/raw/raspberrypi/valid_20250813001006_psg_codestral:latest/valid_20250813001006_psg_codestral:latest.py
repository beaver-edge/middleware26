import cv2
from ai_edge_litert.interpreter import Interpreter
import numpy as np

# setup
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
input_shape  = "data/object_detection/sheeps.mp4"
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f]
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details  = "results/object_detection/test_results/sheeps_detections.mp4"

# open video file
video = cv2.VideoCapture("path_to_your_video")
while(video.isOpened()):
    ret, frame = video.read()
    if not ret:
        break
    # preprocessing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, input_shape)
    input_data = np.expand_dims(resized_frame, axis=0).astype('float32')
    # inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # output interpretation and handling
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    # draw bounding box and label on frame if score is high enough
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * frame.shape[0])))
            xmin = int(max(1,(boxes[i][1] * frame.shape[1])))
            ymax = int(min(frame.shape[0],(boxes[i][2] * frame.shape[0])))
            xmax = int(min(frame.shape[1],(boxes[i][3] * frame.shape[1])))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2)
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, ymin - labelSize[1] - 10), (xmin + labelSize[0], ymin + baseLine - 10), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # display the frame with bounding box and labels
    cv2.imshow('Object detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cleanup
video.release()
cv2.destroyAllWindows()