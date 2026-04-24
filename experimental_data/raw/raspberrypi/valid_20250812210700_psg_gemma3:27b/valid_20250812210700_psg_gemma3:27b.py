import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1: Setup (Loading the model is already done in the provided code)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.) - No preprocessing needed for this example

    # Phase 3: Inference (In this example, we'll just display the frame)
    # In a real application, you would load the model, run inference on the frame,
    # and then process the output.

    # Phase 4.3: Handle Output
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()