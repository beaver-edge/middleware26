#include <Arduino.h>
#include "model.h" // Include the TinyML model header

// Define LED pins
const int ledPin = 13; // Onboard LED for most Arduino boards

void setup() {
  Serial.begin(115200); // Start serial communication at 115200 baud rate
  pinMode(ledPin, OUTPUT); // Set the LED pin as an output
}

void loop() {
  float redValue = analogRead(A0) / 1023.0; // Read Red channel value from A0 pin
  float greenValue = analogRead(A1) / 1023.0; // Read Green channel value from A1 pin
  float blueValue = analogRead(A2) / 1023.0; // Read Blue channel value from A2 pin

  // Prepare input for the model
  float input[3] = {redValue, greenValue, blueValue};

  // Ensure that runInference is declared or defined before use
  int output;

  // Check if runInference function is available and callable
#ifdef runInference
  output = runInference(input);
#else
  Serial.println("Error: runInference function not available.");
  return;
#endif

  // Print the result to Serial Monitor
  Serial.print("Model Output: ");
  Serial.println(output);

  // Control the LED based on the model's prediction
  if (output == 1) {
    digitalWrite(ledPin, HIGH); // Turn on the LED
  } else {
    digitalWrite(ledPin, LOW); // Turn off the LED
  }

  delay(1000); // Wait for a second before next reading
}
