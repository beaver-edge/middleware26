#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "Arduino_APDS9960.h"
#include "model.h"

// Constants for application
const char* CLASSES[] = {"🍎", "🍌", "🍊"};
const int NUM_CLASSES = 3;
const int SERIAL_BAUD_RATE = 9600;

// TensorFlow Lite globals
tflite::MicroErrorReporter error_reporter;
const tflite::Model* tflite_model;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* model_input;
uint8_t tensor_arena[2 * 1024];

// APDS-9960 Sensor
APDS9960 rgbSensor(Wire, 2);  // Adjust the pin number based on your hardware setup

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);

  // Initialize sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize color sensor!");
    while (1);
  }

  // Model loading
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version does not match!");
    while (1);
  }

  // TensorFlow Lite setup
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, sizeof(tensor_arena), &error_reporter);
  interpreter->AllocateTensors();

  // Get model input
  model_input = interpreter->input(0);
}

void loop() {
  int rawRed, rawGreen, rawBlue, proximity;

  // Read from the sensor
  if (!rgbSensor.colorAvailable()) {
    return;
  }
  rgbSensor.readColor(rawRed, rawGreen, rawBlue, proximity);

  // Normalize the sensor data to range [0-1]
  float red_f = rawRed / 255.0;
  float green_f = rawGreen / 255.0;
  float blue_f = rawBlue / 255.0;

  // Fill the input tensor
  model_input->data.f[0] = red_f;
  model_input->data.f[1] = green_f;
  model_input->data.f[2] = blue_f;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Process the output
  TfLiteTensor* output = interpreter->output(0);
  int max_index = 0;
  float max_value = output->data.f[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }

  // Output the recognized class
  Serial.print("Recognized: ");
  Serial.println(CLASSES[max_index]);
  
  delay(1000);  // Add a delay between readings
}
