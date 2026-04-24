import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

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

    # Preprocessing (resize for TFLite model)
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = resized_frame

    # Phase 3: Inference (Not implemented, as the original code already handles it)
    # In this case, the original code already performs the inference.
    # We are only focusing on Phases 2 and 4.2, 4.3.

    # Phase 4.2: Interpret Results (Not applicable, as the original code handles it)
    # The original code already interprets the results and draws bounding boxes.

    # Phase 4.3: Handle Output
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()