#include <Wire.h>
#include <Arduino_HTS221.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Model data
#include "model.h"

// Constants
constexpr int kTensorArenaSize = 2 * 1024;
constexpr int kInputSize = 3;
const char* kClasses[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

// TensorFlow Lite variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* current_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  // Start serial communication
  Serial.begin(9600);

  // Load the model
  current_model = tflite::GetModel(model);
  if (current_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch");
    while (true);
  }

  // Set up the interpreter
  static tflite::MicroInterpreter static_interpreter(
    current_model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true);
  }

  // Get model input (should match dimensions and datatype)
  input = interpreter->input(0);
  if ((input->dims->size != 1) || (input->dims->data[0] != kInputSize) || 
      (input->type != kTfLiteFloat32)) {
    Serial.println("Model input dimension/type mismatch");
    while (true);
  }
}

void loop() {
  // Mock sensor data (replace with actual sensor code)
  float red = 0.556;   // Simulated red value
  float green = 0.222; // Simulated green value
  float blue = 0.200;  // Simulated blue value

  // Copy the RGB values into the model input tensor
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Get the output tensor and read the result
  TfLiteTensor* output = interpreter->output(0);
  int highest_index = 0;
  float highest_value = output->data.f[0];
  
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > highest_value) {
      highest_value = output->data.f[i];
      highest_index = i;
    }
  }

  // Output the class to Serial Monitor
  Serial.print("Detected object: ");
  Serial.println(kClasses[highest_index]);

  // Delay to simulate sensor sampling frequency
  delay(1000); // 1 second delay, adjust as needed
}
