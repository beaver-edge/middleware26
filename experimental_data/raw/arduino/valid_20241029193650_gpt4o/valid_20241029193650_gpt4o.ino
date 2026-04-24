#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Constants
constexpr int kTensorArenaSize = 2 * 1024;
constexpr int kBaudRate = 9600;

// TensorFlow Lite Globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t tensor_arena[kTensorArenaSize];

// APDS9960 sensor
APDS9960 apds(Wire, 2); // Example: using Wire and pin 2 for interrupt

// Object classes
const char* object_classes[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  // Initialize serial communication
  Serial.begin(kBaudRate);

  // Initialize the APDS9960 sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Set up the error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version does not match!");
    while (1);
  }

  // Resolve the operators
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Failed to allocate tensors!");
    while (1);
  }

  // Assign model input and output
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read color data
  int r, g, b;
  if (!apds.readColor(r, g, b)) {
    Serial.println("Failed to read color data!");
    return;
  }

  // Normalize and copy data to input tensor
  input->data.f[0] = r / 255.0;
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Perform inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Inference failed!");
    return;
  }

  // Retrieve the output and display the result
  int max_index = 0;
  for (int i = 1; i < output->dims->data[1]; ++i) {
    if (output->data.f[i] > output->data.f[max_index]) {
      max_index = i;
    }
  }

  // Output the classified object
  Serial.println(object_classes[max_index]);

  // Delay before the next reading
  delay(1000);
}
