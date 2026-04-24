import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Phase 1: Setup (Loading the model is done implicitly in Phase 2)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Assuming the model expects a specific input size, resize the frames accordingly
input_size = (300, 300)  # Example input size, adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, input_size)
    input_data = resized_frame  # The resized frame is the input data

    # Phase 3: Inference (In this case, we're just using OpenCV for processing, so no explicit inference step)

    # Phase 4: Output Interpretation & Handling Loop
    # In this example, we're just writing the processed frame to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()