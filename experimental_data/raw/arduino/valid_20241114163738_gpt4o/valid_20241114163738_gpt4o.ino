#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "./model.h"

// Define the Tensor Arena size and buffer
constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Declare critical variables for TensorFlow Lite
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = tflite::GetModel(model); // Renamed to avoid conflict
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Create an op resolver
tflite::AllOpsResolver resolver;

// APDS-9960 sensor object requires TwoWire and interrupt pin
APDS9960 apds9960(Wire, 2); // Replace '2' with the appropriate interrupt pin

// Classes for classification
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  Serial.begin(9600);

  // Sensor setup
  Wire.begin();
  if (!apds9960.begin()) {
    Serial.println("Error initializing APDS-9960!");
    while (1);
  }
  Serial.println("APDS-9960 initialized!");

  // Load the model
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.", tflite_model->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Define model inputs
  input = interpreter->input(0);
  if (input->dims->size != 3 || input->dims->data[0] != 1 || input->dims->data[1] != 1 || input->dims->data[2] != 3) {
    error_reporter->Report("Model input dimensions mismatch");
    while (1);
  }
}

void loop() {
  int red, green, blue;

  // Collect data from the sensor
  if (!apds9960.readColor(red, green, blue)) { // Correct usage of readColor with int references
    Serial.println("Error reading colors!");
    return;
  }

  // Preprocess: Normalize RGB values to match training data (assuming RGB values are scaled 0-255)
  float red_norm = static_cast<float>(red) / 255.0;
  float green_norm = static_cast<float>(green) / 255.0;
  float blue_norm = static_cast<float>(blue) / 255.0;

  // Copy data to model input buffer
  input->data.f[0] = red_norm;
  input->data.f[1] = green_norm;
  input->data.f[2] = blue_norm;

  // Invoke the interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Retrieve output and determine class
  output = interpreter->output(0);
  int max_index = 0;
  float max_value = output->data.f[0];
  
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }

  // Print out the recognized class
  Serial.print("Detected Object: ");
  Serial.println(classes[max_index]);
  delay(1000);
}
