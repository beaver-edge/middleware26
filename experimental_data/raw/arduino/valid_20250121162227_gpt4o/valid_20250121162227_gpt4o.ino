#include <Arduino.h>

// Define pin numbers for the RGB LED
const int redPin = 9;   // PWM pin for Red channel
const int greenPin = 10; // PWM pin for Green channel
const int bluePin = 11; // PWM pin for Blue channel

void setup() {
  // Initialize serial communication for debugging
  Serial.begin(9600);
  
  // Set the LED pins as output
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
}

void loop() {
  // Example RGB values to set using normalized range (0.0 - 1.0)
  float redValue = 0.562;   // This value is based on the mean of Red from dataset
  float greenValue = 0.237; // This value is based on the mean of Green from dataset
  float blueValue = 0.200;  // This value is based on the mean of Blue from dataset

  // Convert normalized RGB values (0.0 - 1.0) to PWM range (0 - 255)
  int redPWM = map(redValue * 100, 0, 100, 0, 255);
  int greenPWM = map(greenValue * 100, 0, 100, 0, 255);
  int bluePWM = map(blueValue * 100, 0, 100, 0, 255);

  // Set the color of the LED
  analogWrite(redPin, redPWM);
  analogWrite(greenPin, greenPWM);
  analogWrite(bluePin, bluePWM);

  // Debug output to serial monitor
  Serial.print("Red PWM: ");
  Serial.print(redPWM);
  Serial.print(", Green PWM: ");
  Serial.print(greenPWM);
  Serial.print(", Blue PWM: ");
  Serial.println(bluePWM);

  // Delay for a short period before repeating the loop
  delay(1000); // Adjust this delay as needed for your application
}
