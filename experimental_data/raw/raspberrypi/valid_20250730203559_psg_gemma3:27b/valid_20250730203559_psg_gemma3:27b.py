import cv2

# Phase 1: Setup
# 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# 1.3 Load Labels (Conditional) - Not applicable here

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

# 2.2 Preprocess Data - In this case, simply read frames from the video

# Phase 4: Output Interpretation & Handling Loop
# 4.2 Interpret Results - No interpretation needed in this example
# 4.3 Handle Output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()