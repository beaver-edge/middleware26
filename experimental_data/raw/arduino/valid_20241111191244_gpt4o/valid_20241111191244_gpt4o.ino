#include <Wire.h>
#include <mbed.h>
#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>
#include "./model.h"

// Global variables for TensorFlow Lite
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Sensor setup
APDS9960 apds(Wire, -1);

// Classification labels
const char* labels[] = {"Apple", "Banana", "Orange"};

void setup() {
  Serial.begin(9600);

  // Setup for TensorFlow Lite error reporting
  static tflite::MicroErrorReporter tflite_error_reporter;
  error_reporter = &tflite_error_reporter;

  // Load the model
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal "
                           "to supported version %d.",
                           tfl_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Resolver to allow all operators
  static tflite::AllOpsResolver resolver;

  // Setup interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory and check for errors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed.");
    return;
  }

  // Get model input and check its dimensions
  input = interpreter->input(0);
  if (input->dims->size != 2 || input->dims->data[0] != 1 ||
      input->dims->data[1] != 3) {
    error_reporter->Report("Input tensor has incorrect dimensions.");
    return;
  }

  // Ensure input data type is float
  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Input tensor is not float32.");
    return;
  }

  // Initialize color sensor
  if (!apds.begin()) {
    error_reporter->Report("Failed to initialize APDS-9960!");
    while (1);
  }
}

void loop() {
  int red, green, blue;
  
  // Read color and proximity data
  if (!apds.colorAvailable()) {
    Serial.println("Error reading from color sensor!");
    delay(1000);
    return;
  }
    
  apds.readColor(red, green, blue);

  // Scale RGB values to [0, 1] range
  float red_scaled = static_cast<float>(red) / 1024;
  float green_scaled = static_cast<float>(green) / 1024;
  float blue_scaled = static_cast<float>(blue) / 1024;

  // Copy data to input tensor
  input->data.f[0] = red_scaled;
  input->data.f[1] = green_scaled;
  input->data.f[2] = blue_scaled;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke() failed.");
    return;
  }

  // Get the output from the model
  output = interpreter->output(0);

  // Determine the classification
  uint8_t max_index = 0;
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.uint8[i] > output->data.uint8[max_index])
      max_index = i;
  }

  // Print the identified object class
  Serial.print("Detected object: ");
  Serial.println(labels[max_index]);

  delay(1000); // Delay between inferences
}
