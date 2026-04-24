#include <Arduino.h>

// Define the pins for R, G, B sensors
const int redPin = A0;   // Analog pin for Red
const int greenPin = A1; // Analog pin for Green
const int bluePin = A2;  // Analog pin for Blue

void setup() {
  // Start the serial communication
  Serial.begin(9600);
}

void loop() {
  // Read the values from the sensors
  float redValue = analogRead(redPin) / 1023.0;
  float greenValue = analogRead(greenPin) / 1023.0;
  float blueValue = analogRead(bluePin) / 1023.0;

  // Print the values to Serial Monitor
  Serial.print("Red: ");
  Serial.print(redValue);
  Serial.print(", Green: ");
  Serial.print(greenValue);
  Serial.print(", Blue: ");
  Serial.println(blueValue);
  
  // Delay for a short period before the next reading
  delay(1000);
}
