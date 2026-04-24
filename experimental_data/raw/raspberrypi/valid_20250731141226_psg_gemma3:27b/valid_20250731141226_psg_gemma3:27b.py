import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Load Labels
try:
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]
except FileNotFoundError:
    labels = []  # Handle the case where the label file is not found
    print("Label file not found. Using default labels.")

# Load the video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Phase 2: Input Acquisition & Preprocessing Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (224, 224))  # Example resizing
    normalized_frame = resized_frame / 255.0  # Example normalization
    input_data = normalized_frame.astype('float32')
    input_data = input_data[None, ...]  # Add batch dimension

    # Phase 4.2: Interpret Results (Placeholder - Replace with actual interpretation)
    # In a real application, you would perform inference here and interpret the output.
    # For this example, we'll just draw a rectangle on the frame.
    # output_data = interpreter.invoke(input_data)
    # predicted_class = np.argmax(output_data)
    # label = labels[predicted_class] if labels else str(predicted_class)

    # Phase 4.3: Handle Output
    # Draw a rectangle on the frame
    cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)
    # Add label to the frame
    # cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()