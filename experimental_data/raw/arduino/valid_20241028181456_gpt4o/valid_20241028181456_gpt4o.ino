#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <Arduino_APDS9960.h>

// Load the model
#include "model.h"

// Constants
constexpr int kTensorArenaSize = 1024;
constexpr int kNumClasses = 3;
const char* kClasses[kNumClasses] = {"🍎", "🍌", "🍊"};

// Global variables
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter;
const tflite::Model* tflite_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// APDS9960 sensor initialization with default I2C and interrupt pin
APDS9960 rgbSensor(Wire, -1); // Assuming -1 as default interrupt pin if not using interrupts

// Setup function
void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Initialize sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS-9960 sensor!");
    while (1);
  }

  // Set up TensorFlow Lite
  error_reporter = &micro_error_reporter;
  tflite_model = tflite::GetModel(model);
  // Remove version check as TFLITE_SCHEMA_VERSION is not defined
  // if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
  //   error_reporter->Report("Model provided is schema version %d not equal to supported version %d.", tflite_model->version(), TFLITE_SCHEMA_VERSION);
  //   return;
  // }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get input and output tensor pointers
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// Loop function
void loop() {
  // Read RGB data from the sensor
  int r, g, b; // Change to int to match the library's function signature
  if (!rgbSensor.readColor(r, g, b)) {
    Serial.println("Error reading color data");
    return;
  }

  // Normalize the sensor data and copy to the model's input tensor
  input->data.f[0] = r / 255.0; // Assuming 8-bit color depth
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Invoke the interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Process the output
  uint8_t max_index = 0;
  for (int i = 1; i < kNumClasses; ++i) {
    if (output->data.uint8[i] > output->data.uint8[max_index]) {
      max_index = i;
    }
  }

  // Print the result
  Serial.println(kClasses[max_index]);

  // Small delay before next loop
  delay(1000);
}
