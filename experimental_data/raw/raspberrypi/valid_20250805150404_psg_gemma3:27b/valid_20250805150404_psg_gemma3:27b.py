import cv2

# Phase 1: Setup
# 1.2 Paths/Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# 1.3 Load Labels (Conditional)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f]

# 1.4 Load Interpreter (Not needed for cv2 implementation)

# Phase 2: Input Acquisition & Preprocessing Loop
# 2.1 Acquire Input Data
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Phase 4: Output Interpretation & Handling Loop
# 4.2 Interpret Results (Not needed for cv2 implementation)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (300, 300))
    normalized_frame = resized_frame / 255.0

    # Perform inference (replace with your model's inference code)
    # For this example, we'll just add a bounding box around the center of the frame
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    cv2.rectangle(frame, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (0, 255, 0), 2)

    # 4.3 Handle Output
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()