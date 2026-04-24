import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Phase 1: Setup
# Load Labels
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the model
try:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
except Exception as e:
    print(f"Error loading input: {e}")
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

    # Preprocess the frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (224, 224))
    input_data = resized_frame / 255.0  # Normalize to [0, 1]
    input_data = input_data.astype('float32')
    input_data = input_data[None, ...]  # Add batch dimension

    # Phase 3: Inference (Not applicable in this case, as we are using OpenCV for processing)

    # Phase 4.2: Interpret Results (Not applicable in this case, as we are using OpenCV for processing)

    # Phase 4.3: Handle Output
    # Draw bounding boxes and labels on the frame (replace with your actual detection logic)
    # For demonstration purposes, let's just write "Detected" on the frame
    cv2.putText(frame, "Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the output frame to the video file
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()