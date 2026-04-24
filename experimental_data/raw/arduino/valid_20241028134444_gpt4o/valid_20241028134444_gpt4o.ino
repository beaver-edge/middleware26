#include <Arduino.h>

// Constants
const int ledPin = 13; // Pin number for the LED

// Variables
int ledState = LOW;  // Variable to store the current state of the LED

void setup() {
  // Initialize the LED pin as an output
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // Toggle the LED state
  ledState = !ledState;
  
  // Set the LED pin to the current state
  digitalWrite(ledPin, ledState);
  
  // Wait for a second
  delay(1000);
}
