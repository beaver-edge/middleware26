#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Application parameters
const int tensor_arena_size = 2 * 1024;
uint8_t tensor_arena[tensor_arena_size];
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tfl_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// RGB sensor vars
APDS9960 rgbSensor(Wire, -1); // Assuming no interrupt pin is used
int red, green, blue;

// Classification classes
const char* classes[] = {"Apple", "Banana", "Orange"};

// Setup and initialization
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Set up the error reporter
  error_reporter = &micro_error_reporter;

  // Load the model
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version mismatch");
    return;
  }
  
  // AllOpsResolver for compatibility
  static tflite::AllOpsResolver resolver;

  // Build the interpreter
  interpreter = new tflite::MicroInterpreter(tfl_model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  
  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("Tensor allocation failed");
    return;
  }

  // Prepare model input and output
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize the RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor");
    while (1);
  }
}

// Main execution loop
void loop() {
  // Collect data from the RGB sensor
  if (rgbSensor.colorAvailable()) {
    rgbSensor.readColor(red, green, blue);

    // Normalize and copy sensor data to the input tensor
    input->data.f[0] = red / 255.0f;
    input->data.f[1] = green / 255.0f;
    input->data.f[2] = blue / 255.0f;

    // Invoke the model
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Model invocation failed");
      return;
    }

    // Process the output
    uint8_t predicted_class = output->data.uint8[0];
    Serial.print("Predicted Object: ");
    Serial.println(classes[predicted_class]);
  }

  delay(1000);  // Adjust delay as necessary
}
