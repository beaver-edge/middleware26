#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Include the model
#include "model.h"

// Define the Tensor Arena
constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Declare critical variables for TensorFlow Lite Micro
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model_ptr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Instantiate the Sensor with Wire and interrupt pin (modify the intPin as per your setup)
APDS9960 rgbSensor(Wire, -1); // Use -1 if you don't have an interrupt pin

// Classification output classes
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  // Start Serial communication
  Serial.begin(9600);

  // Sensor setup
  Wire.begin();
  if (!rgbSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Set the error reporter to be the default micro error reporter
  static tflite::MicroErrorReporter static_error_reporter;
  error_reporter = &static_error_reporter;

  // Load the model
  model_ptr = tflite::GetModel(model);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version " + String(model_ptr->version()));
    Serial.println("Supported schema version is " + String(TFLITE_SCHEMA_VERSION));
    while (1);
  }

  // Resolve Operators
  tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model_ptr, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's input and output tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read RGB values
  int r, g, b;
  if (!rgbSensor.colorAvailable()) {
    delay(100);
    return;
  }

  rgbSensor.readColor(r, g, b);

  // Preprocessing - Normalize the RGB values
  float input_r = r / 255.0f;
  float input_g = g / 255.0f;
  float input_b = b / 255.0f;

  // Copy data to input tensor
  input->data.f[0] = input_r;
  input->data.f[1] = input_g;
  input->data.f[2] = input_b;

  // Invoke the interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Process the output of model inference
  int class_index = argmax(output->data.uint8, 3);
  Serial.println(classes[class_index]);

  delay(1000);
}

// Function to find the index of the highest value
int argmax(uint8_t* output_data, int length) {
  int max_index = 0;
  uint8_t max_value = output_data[0];
  for (int i = 1; i < length; ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  return max_index;
}
