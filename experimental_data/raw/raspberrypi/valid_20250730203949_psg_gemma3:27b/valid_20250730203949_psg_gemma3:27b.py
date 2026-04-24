import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Phase 1: Setup
# 1.2 Paths/Parameters (already defined above)
# 1.3 Load Labels
try:
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]
except FileNotFoundError:
    labels = []  # Use an empty list if labels file is not found

# 1.4 Load Interpreter (Not needed with cv2)
# 1.5 Get Model Details (Not needed with cv2)

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Phase 4: Output Interpretation & Handling Loop
# 4.1 Get Output Tensor(s) (Not needed with cv2)
# 4.2 Interpret Results (Not needed with cv2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame (replace with your actual object detection/classification code)
    # Example: Draw a rectangle on the frame
    # cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()