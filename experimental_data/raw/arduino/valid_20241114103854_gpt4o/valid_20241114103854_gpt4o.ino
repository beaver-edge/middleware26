#include <Wire.h>
#include <SPI.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "./model.h" // Path to the quantized model header file

// TensorFlow Lite globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr; // Renaming to avoid conflict
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Model input/output dimensions
constexpr int kTensorArenaSize = 1024;
uint8_t tensor_arena[kTensorArenaSize];

// RGB sensor setup
void setupSensor() {
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (true); // Cannot proceed without the sensor
  }
}

// Initialization
void setup() {
  // Set up serial communication
  Serial.begin(9600);
  setupSensor();

  // Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.", tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// Main loop
void loop() {
  // Read RGB values
  int red, green, blue;
  if (!APDS.readColor(red, green, blue)) {
    Serial.println("Error reading color values!");
    return;
  }

  // Preprocessing: Normalize and prepare input data
  input->data.f[0] = red / 255.0f;
  input->data.f[1] = green / 255.0f;
  input->data.f[2] = blue / 255.0f;

  // Perform inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Postprocessing: Map output to object classes
  int class_idx = 0;
  float max_val = 0.0;
  for (int i = 0; i < 3; i++) {
    if (output->data.uint8[i] > max_val) {
      max_val = output->data.uint8[i];
      class_idx = i;
    }
  }

  // Output the result
  switch (class_idx) {
    case 0:
      Serial.println("Detected: 🍏 Apple");
      break;
    case 1:
      Serial.println("Detected: 🍌 Banana");
      break;
    case 2:
      Serial.println("Detected: 🍊 Orange");
      break;
    default:
      Serial.println("Unknown object");
      break;
  }

  delay(1000); // Delay before next reading
}
