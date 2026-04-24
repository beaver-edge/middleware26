#include <Arduino_APDS9960.h>
#include <Wire.h>

// Initialization
APDS9960 apds(Wire, 2); // Assuming pin 2 for interrupt, update as needed

// Define class labels
const char* class_names[] = {"🍎", "🍌", "🍊"};

void setup() {
  // Start serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }
}

void loop() {
  // Preprocessing: Read color sensor values
  int r, g, b, c;
  if (!apds.readColor(r, g, b, c)) {
    Serial.println("Error reading color sensor!");
    return;
  }

  // Normalize RGB values
  float red = r / 255.0;
  float green = g / 255.0;
  float blue = b / 255.0;

  // Placeholder for inference (TensorFlow Lite Micro inference code would go here)
  // Currently, just print normalized values
  Serial.print("Normalized RGB: ");
  Serial.print(red, 4);
  Serial.print(", ");
  Serial.print(green, 4);
  Serial.print(", ");
  Serial.println(blue, 4);

  // Placeholder for classification result
  // Assume class_index is obtained from the model inference
  uint8_t class_index = 0; // Replace with actual inference result

  if (class_index < 3) {
    Serial.print("Detected object: ");
    Serial.println(class_names[class_index]);
  } else {
    Serial.println("Unknown class.");
  }

  delay(1000);
}
