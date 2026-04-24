#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Classification labels
const char* classes[3] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

// Define Tensor Arena for TensorFlow Lite Micro
#define TENSOR_ARENA_SIZE (2 * 1024)
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Setup a logger to capture the machine learning framework errors (MicroErrorReporter)
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Global variables for TensorFlow Lite model objects
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// Model input and output pointers
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Setup TensorFlow Lite
TfLiteStatus setupTensorFlow() {
  tfl_model = tflite::GetModel((const void*)model);  // Use model from model.h
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not supported. Expected %d", tfl_model->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return allocate_status;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  return kTfLiteOk;
}

// Arduino setup function
void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor");
    while (1);
  }

  if (setupTensorFlow() != kTfLiteOk) {
    Serial.println("Error setting up TensorFlow");
    while (1);
  }

  Serial.println("Setup complete");
}

// Arduino loop function
void loop() {
  int red, green, blue;
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  APDS.readColor(red, green, blue);

  float red_norm = (float)red / 255.0;
  float green_norm = (float)green / 255.0;
  float blue_norm = (float)blue / 255.0;

  input->data.f[0] = red_norm;
  input->data.f[1] = green_norm;
  input->data.f[2] = blue_norm;

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  uint8_t max_index = 0;
  for (int i = 1; i < 3; i++) {
    if (output->data.uint8[i] > output->data.uint8[max_index]) {
      max_index = i;
    }
  }

  Serial.println(classes[max_index]);
  delay(1000);
}
