#include <Arduino.h>

// Define pin numbers
const int ledPin = 13; // Pin connected to an LED

// Setup function runs once when you press reset or power the board
void setup() {
  // Initialize the digital pin as an output.
  pinMode(ledPin, OUTPUT);
}

// Loop function runs over and over again forever
void loop() {
  // Turn the LED on
  digitalWrite(ledPin, HIGH);
  delay(1000); // Wait for a second

  // Turn the LED off
  digitalWrite(ledPin, LOW);
  delay(1000); // Wait for a second
}
