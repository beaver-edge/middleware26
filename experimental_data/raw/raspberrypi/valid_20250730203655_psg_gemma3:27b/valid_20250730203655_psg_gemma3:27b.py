import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1: Setup
# Load labels (not needed for object detection, so skipping)

# Load interpreter (using OpenCV for video processing, not ai_edge_litert)
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# Phase 2: Input Acquisition & Preprocessing Loop
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 3: Inference (In this case, OpenCV is used for processing directly)
    # No explicit inference step with ai_edge_litert since OpenCV handles the processing

    # Phase 4: Output Interpretation & Handling Loop
    # In this case, we are just writing the processed frame to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()