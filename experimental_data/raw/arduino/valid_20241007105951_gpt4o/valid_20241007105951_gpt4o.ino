#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "./model.h"  // Make sure this file contains the model data as a byte array

#include <Arduino.h>

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflite_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  
  constexpr int tensor_arena_size = 2 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Initialize APDS-9960 with Wire and interrupt pin (assuming pin 2 for interrupt)
  APDS9960 apds(Wire, 2);
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!apds.begin()) {
    Serial.println("Error initializing APDS-9960 sensor!");
    while (1);
  }
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  tflite_model = tflite::GetModel(model);  // Ensure 'model' is the byte array containing your model
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match TFLite schema version.");
    while (1);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int r, g, b, c;
  if (!apds.readColor(r, g, b, c)) {
    Serial.println("Error reading color sensor data!");
    return;
  }

  input->data.f[0] = static_cast<float>(r) / 1024.0f;
  input->data.f[1] = static_cast<float>(g) / 1024.0f;
  input->data.f[2] = static_cast<float>(b) / 1024.0f;

  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  uint8_t max_index = 0;
  for (int i = 1; i < 3; i++) {
    if (output->data.uint8[i] > output->data.uint8[max_index]) {
      max_index = i;
    }
  }

  switch (max_index) {
    case 0:
      Serial.println("🍎 Apple");
      break;
    case 1:
      Serial.println("🍌 Banana");
      break;
    case 2:
      Serial.println("🍊 Orange");
      break;
    default:
      Serial.println("Unknown");
      break;
  }

  delay(1000);
}
