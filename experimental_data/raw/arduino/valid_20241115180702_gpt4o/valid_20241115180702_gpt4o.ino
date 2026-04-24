#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Declare critical TensorFlow Lite Micro variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Define tensor arena size and allocate buffer
constexpr int kTensorArenaSize = 1024;
uint8_t tensor_arena[kTensorArenaSize];

// RGB sensor with default wire and interrupt pin
APDS9960 apds(Wire, 2);

// Class labels
const char* class_names[] = {"Apple", "Banana", "Orange"};

void setup() {
  Serial.begin(9600);

  // Initialize the color sensor
  if (!apds.begin()) {
    error_reporter->Report("Failed to initialize APDS9960!");
    while (1);
  }
  
  // Load the model
  tflite_model = tflite::GetModel((const unsigned char*)model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema!");
    while (1);
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("Tensor allocation failed!");
    while (1);
  }

  // Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Setup Completed Successfully.");
}

void loop() {
  // Read RGB values from the sensor
  int r, g, b, p;
  if (apds.colorAvailable() && apds.readColor(r, g, b, p)) {
    // Normalize the sensor data
    input->data.f[0] = r / 65535.0f;
    input->data.f[1] = g / 65535.0f;
    input->data.f[2] = b / 65535.0f;

    // Invoke interpreter
    if (interpreter->Invoke() != kTfLiteOk) {
      error_reporter->Report("Invoke failed!");
      while (1);
    }

    // Process the output of model inference
    int max_index = 0;
    uint8_t max_value = output->data.uint8[0];
    for (int i = 1; i < 3; i++) {
      if (output->data.uint8[i] > max_value) {
        max_value = output->data.uint8[i];
        max_index = i;
      }
    }
    
    // Output the result
    Serial.print("Detected Object: ");
    Serial.println(class_names[max_index]);
  }
  
  delay(1000);
}
