#include <Arduino.h>

// Define a global array to store the output data for RGB values
float output_data[3] = {0.0, 0.0, 0.0};

void setup() {
    // Initialize serial communication at 9600 baud rate
    Serial.begin(9600);
}

void loop() {
    // Example of assigning some values to output_data (this could be replaced with actual model inference)
    output_data[0] = 0.5;   // Red value
    output_data[1] = 0.25;  // Green value
    output_data[2] = 0.2;   // Blue value

    for(int i = 0; i < 3; i++) {
        Serial.print("Value ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(output_data[i], 6); // Adjust the decimal places as needed
    }
    
    delay(1000); // Add a delay to avoid spamming too many prints per second
}
