#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"  // Include the model header to access model data

// Communication settings
#define SERIAL_BAUD_RATE 9600

// Color sensor instance
APDS9960 apds(Wire, -1);  // correct constructor with TwoWire reference and no interrupt pin

// TensorFlow Lite Globals
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

void setup() {
  // Start serial communication
  Serial.begin(SERIAL_BAUD_RATE);

  // Initialize the error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model from the model data
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model schema version does not match.");
    return;
  }

  // Set up the op resolver
  static tflite::AllOpsResolver resolver;

  // Set up the interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed.");
    return;
  }

  // Get input tensor
  input = interpreter->input(0);

  // Initialize the color sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS-9960 sensor!");
    while (1);
  }
}

void loop() {
  // Read RGB values from the sensor
  int r, g, b;
  if (!apds.colorAvailable() || !apds.readColor(r, g, b)) {
    Serial.println("Error reading color!");
    return;
  }

  // Normalize the RGB values and copy to input tensor
  input->data.f[0] = static_cast<float>(r) / 65535.0;
  input->data.f[1] = static_cast<float>(g) / 65535.0;
  input->data.f[2] = static_cast<float>(b) / 65535.0;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    return;
  }

  // Obtain the result from the output tensor
  TfLiteTensor* output = interpreter->output(0);
  uint8_t class_index = output->data.uint8[0];

  // Output the classification result with emojis
  switch (class_index) {
    case 0:
      Serial.println("Apple 🍎");
      break;
    case 1:
      Serial.println("Banana 🍌");
      break;
    case 2:
      Serial.println("Orange 🍊");
      break;
    default:
      Serial.println("Unknown 🤔");
      break;
  }

  delay(1000);  // Add delay to not flood the Serial output
}
