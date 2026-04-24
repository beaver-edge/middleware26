import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1: Setup (Loading model and setting up video capture)
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Phase 2: Input Acquisition & Preprocessing Loop
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing: Resize and normalize the frame
    resized_frame = cv2.resize(frame, (300, 300))
    normalized_frame = resized_frame / 255.0

    # Convert to numpy array and expand dimensions
    input_data = normalized_frame.astype('float32')
    input_data = input_data[None, :, :, :]  # Add batch dimension

    # Phase 3: Inference (Not applicable as we are using OpenCV directly)

    # Phase 4.2: Interpret Results (Not applicable as we are using OpenCV directly)

    # Phase 4.3: Handle Output (Draw bounding boxes on the frame)
    # In this example, we'll just write the processed frame to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()