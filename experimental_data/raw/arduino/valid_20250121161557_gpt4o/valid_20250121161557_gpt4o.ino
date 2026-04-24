#include <Arduino.h>

// Define the pins for RGB LED
const int redPin = 9;   // Red pin connected to PWM output
const int greenPin = 10; // Green pin connected to PWM output
const int bluePin = 11; // Blue pin connected to PWM output

void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);

  Serial.begin(9600); // Start serial communication for debugging purposes
}

void loop() {
  // Assuming we have a function `getAverageColor` that returns the average color values
  float avgRed = 0.5623661971830984;   // Placeholder: Replace with actual data retrieval logic
  float avgGreen = 0.23716901408450702; // Placeholder: Replace with actual data retrieval logic
  float avgBlue = 0.200056338028169;    // Placeholder: Replace with actual data retrieval logic

  // Map the average color values to PWM range (0-255)
  int redValue = map(avgRed * 100, 0, 100, 0, 255);
  int greenValue = map(avgGreen * 100, 0, 100, 0, 255);
  int blueValue = map(avgBlue * 100, 0, 100, 0, 255);

  // Set the RGB LED color
  analogWrite(redPin, redValue);
  analogWrite(greenPin, greenValue);
  analogWrite(bluePin, blueValue);

  delay(500); // Adjust as needed for demonstration purposes

  // Debugging: Print current values to serial monitor
  Serial.print("Red: ");
  Serial.print(avgRed);
  Serial.print(", Green: ");
  Serial.print(avgGreen);
  Serial.print(", Blue: ");
  Serial.println(avgBlue);
}
