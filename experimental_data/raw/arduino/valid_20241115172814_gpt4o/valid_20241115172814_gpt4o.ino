#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
// #include "model_data.h" - Commented out because this file path is missing.
// Ensure you have the correct model header file available and uncomment this line.

// Define Constants
const int kTensorArenaSize = 2 * 1024; // 2KB Tensor Arena

// Globals for TensorFlow Lite Micro
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
uint8_t tensor_arena[kTensorArenaSize];
TfLiteTensor* input_tensor = nullptr;

// Constants for RGB sensor reading
int red, green, blue;

void setup() {
  // Initialize serial
  Serial.begin(9600);
  while (!Serial) {}  // Wait for serial port to connect

  // Initialize TFLite model
  // model = tflite::GetModel(g_model);  // Placeholder: Correct model data must be defined
  // You need to ensure that the correct model data header file is included and defined properly.
  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   error_reporter->Report("Model version does not match Schema");
  //   return;
  // }

  // Define ops resolver and set up the interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get input tensor pointer
  input_tensor = interpreter->input(0);
  if (input_tensor->dims->data[0] != 1 || input_tensor->dims->data[1] != 3) {
    error_reporter->Report("Input tensor has incorrect dimensions");
    return;
  }

  // Initialize APDS9960 (Color Sensor)
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS-9960 sensor!");
    return;
  }
  APDS.setGestureSensitivity(0);
}

void loop() {
  // Read color data from APDS9960
  if (!APDS.colorAvailable()) {
    delay(5); // Wait for color data
    return;
  }

  APDS.readColor(red, green, blue);

  // Normalize the RGB values between 0 and 1
  float normalized_red = red / 255.0;
  float normalized_green = green / 255.0;
  float normalized_blue = blue / 255.0;

  // Copy normalized RGB values to model input
  input_tensor->data.f[0] = normalized_red;
  input_tensor->data.f[1] = normalized_green;
  input_tensor->data.f[2] = normalized_blue;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Process the model output
  TfLiteTensor* output = interpreter->output(0);
  uint8_t class_idx = output->data.uint8[0];

  // Output object class based on index
  switch (class_idx) {
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

  delay(1000); // Delay between inferences
}
