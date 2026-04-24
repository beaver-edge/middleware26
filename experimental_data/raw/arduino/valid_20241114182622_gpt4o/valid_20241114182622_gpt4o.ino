#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Include the model
#include "model.h"

// Classes
const char* classes[] = {"Apple", "Banana", "Orange"};

// TensorFlow Lite globals
tflite::MicroErrorReporter errorReporter;
const tflite::Model* tfModel = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensorArena[kTensorArenaSize];

// Sensor
APDS9960 rgbSensor(Wire, -1);  // Assuming no interrupt pin used

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  // Setup the sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }

  // Load the model
  tfModel = tflite::GetModel(model);
  if (tfModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version not supported!");
    while (1);
  }

  // Create the AllOpsResolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(tfModel, resolver, tensorArena, kTensorArenaSize, &errorReporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor arena
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read RGB values
  int r, g, b;
  if (!rgbSensor.readColor(r, g, b)) {
    Serial.println("Error reading RGB values!");
    return;
  }

  // Normalize and populate input tensor
  input->data.f[0] = r / 255.0f;
  input->data.f[1] = g / 255.0f;
  input->data.f[2] = b / 255.0f;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Get prediction result
  uint8_t predicted_class = 0;
  uint8_t max_score = output->data.uint8[0];
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.uint8[i] > max_score) {
      max_score = output->data.uint8[i];
      predicted_class = i;
    }
  }

  // Display result with Unicode emojis
  Serial.print("Detected: ");
  switch (predicted_class) {
    case 0:
      Serial.println("Apple 🍏");
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
