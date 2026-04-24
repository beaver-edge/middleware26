#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "./model.h"

// Constants
const int kTensorArenaSize = 2 * 1024;
const char* kClasses[] = {"Apple", "Banana", "Orange"};

// Global Variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tf_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
uint8_t tensor_arena[kTensorArenaSize];

// Sensor
APDS9960 rgbSensor(Wire, 2); // Assuming Pin 2 for interrupt

// Setup function
void setup() {
  Serial.begin(9600);

  // Initialize sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }

  // Load the model
  tf_model = tflite::GetModel(model);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while (1);
  }

  // Specify the ops resolver
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      tf_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// Loop function
void loop() {
  // Read RGB values from the sensor
  int r, g, b;
  if (!rgbSensor.readColor(r, g, b)) {
    Serial.println("Failed to read color data");
    return;
  }

  // Normalize and copy the data to the model's input tensor
  input->data.f[0] = r / 65536.0;
  input->data.f[1] = g / 65536.0;
  input->data.f[2] = b / 65536.0;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Output the result
  uint8_t class_index = output->data.uint8[0];
  Serial.println(kClasses[class_index]);

  delay(1000);
}
