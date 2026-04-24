import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"
label_path  = "models/ssd-mobilenet_v1/labelmap.txt"

# Load Labels
try:
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]
except FileNotFoundError:
    labels = []  # Handle case where label file is not found

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

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, normalize, etc.) - Placeholder for actual preprocessing
    # Replace this with the actual preprocessing steps required by your model
    resized_frame = cv2.resize(frame, (224, 224))  # Example resize
    normalized_frame = resized_frame / 255.0  # Example normalization
    input_data = normalized_frame.reshape(1, 224, 224, 3)

    # Perform inference
    # Placeholder for model inference
    # Replace this with your actual inference code
    # Assuming the model outputs class probabilities
    # For demonstration, we'll use a dummy output
    dummy_output = [0.1, 0.2, 0.7]  # Example probabilities for 3 classes
    predicted_class = dummy_output.index(max(dummy_output))
    predicted_label = labels[predicted_class] if labels else str(predicted_class)

    # Add label to frame
    text = f"Predicted: {predicted_label}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()