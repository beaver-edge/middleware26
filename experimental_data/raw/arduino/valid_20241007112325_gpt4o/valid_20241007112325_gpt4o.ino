#include <Arduino_APDS9960.h>
#include "model.h"

// Initialization: Declare Variables
APDS9960 apds(Wire, -1); // Use default Wire and no interrupt pin

// Initialization: Load the Model
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Sensor Setup
  if (!apds.begin()) {
    Serial.println("Failed to initialize APDS9960!");
    while (1);
  }
}

void loop() {
  // Preprocessing: Sensor Setup
  int r, g, b, c;
  if (!apds.colorAvailable()) {
    delay(5);
    return;
  }
  apds.readColor(r, g, b, c);

  // Normalize sensor readings
  float red = r / 255.0;
  float green = g / 255.0;
  float blue = b / 255.0;

  // Placeholder for inference logic
  // This is where you would integrate your model inference code
  // For now, we'll simulate the output with dummy logic

  // Dummy logic to simulate model output
  int max_index = 0;
  if (red > green && red > blue) {
    max_index = 0; // Apple
  } else if (green > red && green > blue) {
    max_index = 1; // Banana
  } else {
    max_index = 2; // Orange
  }

  // Output classification result
  switch (max_index) {
    case 0:
      Serial.println("Apple 🍎");
      break;
    case 1:
      Serial.println("Banana 🍌");
      break;
    case 2:
      Serial.println("Orange 🍊");
      break;
    default:
      Serial.println("Unknown");
  }

  delay(1000);  // Delay for readability
}
