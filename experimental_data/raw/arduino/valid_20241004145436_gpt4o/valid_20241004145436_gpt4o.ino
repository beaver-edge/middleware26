#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <Wire.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>xxxxxx3

// TensorFlow Lite globals for microcontroller
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model_instance = nullptr; // Avoid conflicting with model array
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Load the model from the model.h file
#include "model.h"

// APDS9960 sensor instance
APDS9960 apds(Wire, /*intPin=*/2);

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Initialize the sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1); // Halt
  }

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model_instance = tflite::GetModel(model); // Use the correct model array
  if (model_instance->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while (1); // Halt
  }

  // Pull in only the operation implementations we need
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model_instance, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1); // Halt
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read RGB values from the sensor
  int r, g, b;
  if (!apds.readColor(r, g, b)) {
    Serial.println("Error reading color!");
    return;
  }

  // Normalize the RGB values based on the dataset summary
  float normalized_red = r / 255.0;
  float normalized_green = g / 255.0;
  float normalized_blue = b / 255.0;

  // Copy the normalized data into the input tensor
  input->data.f[0] = normalized_red;
  input->data.f[1] = normalized_green;
  input->data.f[2] = normalized_blue;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Process the output
  uint8_t predicted_class = output->data.uint8[0];
  if (predicted_class == 0) {
    Serial.println("🍎 Apple");
  } else if (predicted_class == 1) {
    Serial.println("🍌 Banana");
  } else if (predicted_class == 2) {
    Serial.println("🍊 Orange");
  } else {
    Serial.println("Unknown");
  }

  // Delay for a while before the next reading
  delay(1000);
}
