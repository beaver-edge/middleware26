import cv2
import numpy as np

def preprocess_image(image):
    # Resize the image to (300, 300)
    resized_image = cv2.resize(image, (300, 300))

    # Normalize the pixel values to [0, 1] if required by the model
    normalized_image = resized_image / 255.0

    # Expand dimensions to match the expected input shape of the model
    input_data = np.expand_dims(normalized_image, axis=0)

    return input_data