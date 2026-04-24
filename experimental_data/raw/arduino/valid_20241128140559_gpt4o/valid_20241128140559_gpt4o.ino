#include <Arduino.h>

// Define pin numbers
const int redPin = A0;
const int greenPin = A1;
const int bluePin = A2;

// Variables to store sensor readings
float redValue;
float greenValue;
float blueValue;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  // Initialize analog pins
  pinMode(redPin, INPUT);
  pinMode(greenPin, INPUT);
  pinMode(bluePin, INPUT);
}

void loop() {
  // Read the values from the analog pins
  redValue = analogRead(redPin) / 1023.0; // Normalize to [0, 1]
  greenValue = analogRead(greenPin) / 1023.0; // Normalize to [0, 1]
  blueValue = analogRead(bluePin) / 1023.0; // Normalize to [0, 1]

  // Output the values to Serial Monitor
  Serial.print("Red: ");
  Serial.print(redValue);
  Serial.print(", Green: ");
  Serial.print(greenValue);
  Serial.print(", Blue: ");
  Serial.println(blueValue);

  // Add a delay to prevent flooding the Serial Monitor
  delay(1000);
}
