#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// Declare TensorFlow Lite Micro variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Tensor arena for memory allocation
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Initialize APDS9960 sensor with Wire interface and no interrupt pin
APDS9960 apds(Wire, -1);

// Classification labels
const char* class_names[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  // Set up Serial communication
  Serial.begin(9600);

  // Initialize the APDS9960 sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal "
                           "to supported version %d.",
                           tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Set up the all ops resolver
  static tflite::AllOpsResolver resolver;

  // Initialize the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensor memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the input tensor
  input_tensor = interpreter->input(0);
}

void loop() {
  int red, green, blue;

  // Check if color data is available
  if (apds.colorAvailable()) {
    // Read the sensor data
    if (!apds.readColor(red, green, blue)) {
      Serial.println("Error reading color data!");
      return;
    }

    // Normalize the input data by converting them into 0-1 range
    input_tensor->data.f[0] = static_cast<float>(red) / 255.0;
    input_tensor->data.f[1] = static_cast<float>(green) / 255.0;
    input_tensor->data.f[2] = static_cast<float>(blue) / 255.0;

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      error_reporter->Report("Invoke failed.");
      return;
    }

    // Obtain the output from the model
    output_tensor = interpreter->output(0);

    // Determine the classification result
    uint8_t maxIndex = 0;
    for (uint8_t i = 1; i < 3; i++) {
      if (output_tensor->data.uint8[i] > output_tensor->data.uint8[maxIndex]) {
        maxIndex = i;
      }
    }

    // Output the classification
    Serial.println(class_names[maxIndex]);

    // Wait some time before the next sample
    delay(1000);
  }
}
