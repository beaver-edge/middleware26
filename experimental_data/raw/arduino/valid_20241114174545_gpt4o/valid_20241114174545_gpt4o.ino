#include <Arduino.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>
#include "model.h"  // Include the header for your pre-trained model data

#define TENSOR_ARENA_SIZE 2048

// Declare critical variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::AllOpsResolver resolver;

// Define classes
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  // Initialize Serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize the APDS-9960 sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.",
                           tflite_model->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }

  // Instantiate the MicroInterpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Define model inputs and outputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int r, g, b;
  if (!APDS.colorAvailable()) {
    return;  // Return if no color value is available
  }

  // Read RGB values from the sensor
  APDS.readColor(r, g, b);

  // Scale RGB readings to 0-1 range
  input->data.f[0] = r / 1024.0f;
  input->data.f[1] = g / 1024.0f;
  input->data.f[2] = b / 1024.0f;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Process inference output
  uint8_t max_index = 0;
  float max_value = output->data.f[0];
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }

  // Output the classification result
  Serial.println(classes[max_index]);

  // Delay before next inference
  delay(1000);
}
