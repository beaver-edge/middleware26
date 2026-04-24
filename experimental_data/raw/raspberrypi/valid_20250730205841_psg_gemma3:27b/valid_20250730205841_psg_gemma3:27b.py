import cv2

# CONFIGURATION PARAMETERS
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1: Setup (already done with config parameters)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Phase 3: Inference (integrated within the loop)

# Phase 4: Output Interpretation & Handling Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.) - assuming the model expects a specific input size
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = resized_frame / 255.0  # Normalize to [0, 1]
    input_data = input_data.astype('float32')
    input_data = input_data[None, :, :, :]  # Add batch dimension

    # In this case, it's assumed the model will perform object detection or segmentation.
    # The following is a placeholder to show how output would be handled.

    # Assuming model gives bounding boxes and class ids. Replace with actual inference.
    # For demonstration purposes, we'll just draw a rectangle in the center of the frame.
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    cv2.rectangle(frame, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()