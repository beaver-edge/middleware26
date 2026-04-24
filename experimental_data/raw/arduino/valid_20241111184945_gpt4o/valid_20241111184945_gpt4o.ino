#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"  // Ensure that you include the correct model header file

// Constants
#define TENSOR_ARENA_SIZE 1024
const float kInputMean = 0.0;
const float kInputStd = 1.0;
const char *kClasses[] = {"🅰️ Apple", "🍌 Banana", "🍊 Orange"};

// Global variables for TensorFlow Lite
namespace {
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *tflite_model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

// Setup function
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }

  // Set up the error reporter, model, and interpreter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  tflite_model = tflite::GetModel(model);  // Ensure the correct model variable is used
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

// Loop function
void loop() {
  // Read sensor values
  int r, g, b;
  if (!APDS.colorAvailable()) {
    return;
  }
  APDS.readColor(r, g, b);

  // Normalize sensor values to feed into the model input tensor
  input->data.f[0] = (r - kInputMean) / kInputStd;
  input->data.f[1] = (g - kInputMean) / kInputStd;
  input->data.f[2] = (b - kInputMean) / kInputStd;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Get the result of the model inference
  uint8_t best_index = 0;
  float best_score = output->data.f[0];
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.f[i] > best_score) {
      best_index = i;
      best_score = output->data.f[i];
    }
  }

  // Print the result
  Serial.println(kClasses[best_index]);
  delay(1000);
}
