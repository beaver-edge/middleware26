#include <Arduino.h>

// Define pin numbers
const int ledPin = 13;

// Setup function runs once at startup
void setup() {
  // Initialize the LED pin as an output
  pinMode(ledPin, OUTPUT);
}

// Loop function runs repeatedly
void loop() {
  // Turn the LED on
  digitalWrite(ledPin, HIGH);
  // Wait for a second
  delay(1000);
  // Turn the LED off
  digitalWrite(ledPin, LOW);
  // Wait for a second
  delay(1000);
}
