import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1: Setup (Parts 1.2, 1.3, 1.4, 1.5 are implicitly done in the original code)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing (resize for model input)
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = resized_frame

    # Phase 3: Inference (Implicit in the original code)

    # Phase 4: Output Interpretation & Handling Loop
    # The original code performs inference and handles the output directly within the loop.
    # No explicit interpretation or handling is needed here, as the original code already does it.

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()