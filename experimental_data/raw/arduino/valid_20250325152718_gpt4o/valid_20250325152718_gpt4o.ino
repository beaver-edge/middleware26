#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>

#include "model.h"

#define TENSOR_ARENA_SIZE 2 * 1024
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;
const tflite::Model *model_instance;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;

APDS9960 apds(Wire, 2); // Assuming 2 is the interrupt pin
int red, green, blue, clear;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!apds.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }

  model_instance = tflite::GetModel(model);
  if (model_instance->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version not supported.");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  interpreter = new tflite::MicroInterpreter(model_instance, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter->AllocateTensors();

  input = interpreter->input(0);
}

void loop() {
  if (!apds.readColor(red, green, blue, clear)) {
    Serial.println("Error reading color values!");
    return;
  }

  float normRed = red / 255.0;
  float normGreen = green / 255.0;
  float normBlue = blue / 255.0;

  input->data.f[0] = normRed;
  input->data.f[1] = normGreen;
  input->data.f[2] = normBlue;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  TfLiteTensor *output = interpreter->output(0);
  uint8_t result = output->data.uint8[0];

  switch (result) {
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