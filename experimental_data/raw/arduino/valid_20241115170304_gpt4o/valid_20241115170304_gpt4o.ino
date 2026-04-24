#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model.h"

// Declare Variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define Tensor Arena
constexpr int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];

// Set up APDS9960 sensor
APDS9960 apds(Wire, /*Specify appropriate interrupt pin here*/ -1);

void setup() {
  Serial.begin(9600);
  
  // Set up the sensor
  Wire.begin();
  if (!apds.begin()) {
    Serial.println("APDS9960 setup failed");
    while (1);
  }
  
  // Load the Model
  tflite_model = tflite::GetModel(model);

  // Define All Ops Resolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter;

  // Allocate Memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (1);
  }

  // Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Sensor Setup and Feature Extraction
  int r, g, b;
  if (!apds.readColor(r, g, b)) {
    Serial.println("Failed to read color");
    return;
  }

  float red = r / 1024.0;
  float green = g / 1024.0;
  float blue = b / 1024.0;

  // Data Copy
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // Invoke the Interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke tflite interpreter");
    return;
  }
  
  // Process Model Output
  int class_index = 0;
  float max_score = output->data.f[0];
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > max_score) {
      max_score = output->data.f[i];
      class_index = i;
    }
  }

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
  }
  
  delay(1000);
}
