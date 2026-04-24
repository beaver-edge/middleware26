import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Phase 1: Setup (Loading model and setting up video capture)
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Phase 2: Input Acquisition & Preprocessing Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing: Resize and normalize the frame
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = resized_frame / 255.0  # Normalize to [0, 1]
    input_data = input_data.astype('float32')
    input_data = input_data[None, :, :, :]  # Add batch dimension

    # Phase 3: Inference (Placeholder - Replace with actual TFLite inference code)
    # In a real implementation, you would load the TFLite model,
    # allocate tensors, set input, invoke, and get output here.
    # For this example, we'll just copy the input frame as the output.
    output_data = input_data

    # Phase 4: Output Interpretation & Handling Loop
    # In this example, we'll just display the output frame.
    processed_frame = (output_data[0] * 255).astype('uint8')

    # Phase 4.3: Handle Output
    out.write(processed_frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()