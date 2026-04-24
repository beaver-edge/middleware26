#include <Arduino.h>
#include <Wire.h>
#include <TensorFlowLite.h> // Include the base TensorFlow Lite Library
#include "model.h" // Include model header file for the TFLite model binary

// Include dependent TensorFlow Lite Micro header files
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Define tensor arena size
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Declare TensorFlow Lite Micro variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tf_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Load the model
  tf_model = tflite::GetModel(model);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version is not compatible!");
    while (true);
  }

  // Resolve necessary operators
  static tflite::AllOpsResolver micro_op_resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tf_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (true);
  }

  // Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Sensor setup: Initialize RGB sensor (Assume sensors are connected to A0, A1, and A2 analog pins)
  // Replace with the actual initialization for the RGB sensor if different
}

void loop() {
  // Get RGB sensor readings (Replace with actual readings from the RGB sensor)
  float red_value = analogRead(A0) / 1023.0;   // Placeholder for actual Red value reading
  float green_value = analogRead(A1) / 1023.0; // Placeholder for actual Green value reading
  float blue_value = analogRead(A2) / 1023.0;  // Placeholder for actual Blue value reading

  // Copy the data to model input buffer
  input->data.f[0] = red_value;
  input->data.f[1] = green_value;
  input->data.f[2] = blue_value;

  // Invoke the interpreter for inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Process output
  int max_index = 0;
  uint8_t max_value = output->data.uint8[0];
  for (int i = 1; i < 3; i++) {
    if (output->data.uint8[i] > max_value) {
      max_value = output->data.uint8[i];
      max_index = i;
    }
  }

  // Interpret result based on the application specification
  switch (max_index) {
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
