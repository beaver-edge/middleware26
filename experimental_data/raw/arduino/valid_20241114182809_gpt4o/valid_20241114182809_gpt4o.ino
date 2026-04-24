#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// TensorFlow Lite model variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena size
constexpr int kTensorArenaSize = 2 * 1024;
byte tensor_arena[kTensorArenaSize];

// Classes
const char* classes[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the APDS9960 sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Set up the resolver
  static tflite::AllOpsResolver resolver;

  // Create an interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Define RGB variables
  int red, green, blue;

  // Read from the sensor
  if (!APDS.colorAvailable()) return;
  APDS.readColor(red, green, blue);

  // Preprocess the color values to the required float format
  float normalized_red = static_cast<float>(red) / 1024;
  float normalized_green = static_cast<float>(green) / 1024;
  float normalized_blue = static_cast<float>(blue) / 1024;

  // Copy the processed input data to the model input tensor
  input->data.f[0] = normalized_red;
  input->data.f[1] = normalized_green;
  input->data.f[2] = normalized_blue;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error invoking the interpreter");
    return;
  }

  // Postprocess the output
  int highest_index = -1;
  float highest_score = 0.0;
  for (int i = 0; i < 3; i++) {
    float score = output->data.f[i];
    if (score > highest_score) {
      highest_score = score;
      highest_index = i;
    }
  }

  // Output the result
  if (highest_index != -1) {
    Serial.print("Detected: ");
    Serial.println(classes[highest_index]);
  }

  delay(1000);  // Delay between readings
}
