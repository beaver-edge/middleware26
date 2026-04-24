#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h" // Your TensorFlow Lite model

// TensorFlow Lite globals
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter;

  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Create an area of memory to use for input, output, and intermediate arrays
  constexpr int kTensorArenaSize = 2 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(9600);

  // Initialize the APDS9960 sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Load the model
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version isn't compatible!");
    while (1);
  }

  // Use the MicroMutableOpResolver to load needed operators
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddDequantize();

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's input and output tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    while (1);
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read color values
  int r, g, b, c;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, c);

    // Normalize sensor values and write them to the model's input tensor
    input->data.f[0] = r / 255.0;
    input->data.f[1] = g / 255.0;
    input->data.f[2] = b / 255.0;

    // Run the model on the input
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Get the classification result from the model's output tensor
    int class_index = output->data.uint8[0];

    // Output the object class to the serial monitor
    const char* class_names[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};
    Serial.print("Detected: ");
    Serial.println(class_names[class_index]);
    
    delay(1000);
  }
}
