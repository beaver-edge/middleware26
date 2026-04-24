import cv2

# Configuration Parameters
model_path  = "models/ssd-mobilenet_v1/detect.tflite"
input_path  = "data/object_detection/sheeps.mp4"
output_path  = "results/object_detection/test_results/sheeps_detections.mp4"

# Phase 1: Setup (Model loading is already done in the original script)

# Phase 2: Input Acquisition & Preprocessing Loop
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Phase 3: Inference & Phase 4: Output Interpretation & Handling Loop
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to the expected input size (assuming 300x300)
    resized_frame = cv2.resize(frame, (300, 300))
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize the frame (assuming values between 0 and 255)
    input_data = (rgb_frame / 255.0).astype('float32')
    
    # Reshape the input data to match the model's expected input shape (assuming (1, 300, 300, 3))
    input_data = input_data.reshape((1, 300, 300, 3))

    # Inference (Placeholder - Replace with actual TFLite inference code)
    # Assuming a dummy output for demonstration
    # output_data = interpreter.invoke(input_data)

    # Replace the following with actual interpretation based on the model output
    # Dummy output for demonstration: bounding boxes and labels
    # output_data = [[[100, 100, 200, 200, 0.9, 0], [50, 50, 150, 150, 0.8, 1]]]
    
    # Drawing bounding boxes (Replace with actual interpretation)
    # for box in output_data[0]:
    #     x1, y1, x2, y2, confidence, class_id = box
    #     x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, f"Class: {class_id}, Conf: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

# Phase 5: Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()