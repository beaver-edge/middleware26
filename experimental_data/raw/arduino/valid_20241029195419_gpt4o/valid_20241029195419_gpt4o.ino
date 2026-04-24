#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

// Initialization
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Load the model
const tflite::Model* tflite_model = tflite::GetModel(model);

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Object Classifier by Color");

  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema!");
    while (1);
  }

  // Resolve Operators
  tflite::AllOpsResolver resolver;

  // Tensor Arena
  constexpr int kTensorArenaSize = 2048;
  static uint8_t tensor_arena[kTensorArenaSize];

  // Interpreter
  static tflite::MicroInterpreter interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate Memory
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Sensor setup
  APDS9960 apds(Wire, -1); // Assuming no interrupt pin used
  bool success = apds.begin();
  if (!success) {
    error_reporter->Report("Failed to initialize APDS9960 sensor!");
    while (1);
  }
}

void loop() {
  static APDS9960 apds(Wire, -1);
  static tflite::MicroInterpreter* interpreter = nullptr;
  static TfLiteTensor* input = nullptr;
  static TfLiteTensor* output = nullptr;

  if (interpreter == nullptr) {
    // Resolve Operators
    tflite::AllOpsResolver resolver;

    // Tensor Arena
    constexpr int kTensorArenaSize = 2048;
    static uint8_t tensor_arena[kTensorArenaSize];

    // Interpreter
    static tflite::MicroInterpreter local_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &local_interpreter;

    // Allocate Memory
    interpreter->AllocateTensors();

    // Input and Output
    input = interpreter->input(0);
    output = interpreter->output(0);
  }

  int r, g, b;
  
  // Read RGB values from sensor
  if (!apds.colorAvailable()) {
    return;
  }
  apds.readColor(r, g, b);

  // Preprocessing: Normalize the RGB values to 0-1 range
  input->data.f[0] = r / 65535.0f;
  input->data.f[1] = g / 65535.0f;
  input->data.f[2] = b / 65535.0f;

  // Inference: Invoke the interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed!");
    return;
  }

  // Postprocessing: Process the output
  uint8_t class_index = output->data.uint8[0];
  switch (class_index) {
    case 0:
      Serial.println("🍏 Apple");
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

  delay(1000); // Wait for a second before next reading
}
