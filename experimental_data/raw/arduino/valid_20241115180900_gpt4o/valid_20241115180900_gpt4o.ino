#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>   
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Wire.h>
#include "model.h"

// Define the tensor arena size
const int tensorArenaSize = 2048;
uint8_t tensorArena[tensorArenaSize];

// Declare critical variables for TensorFlow Lite Micro
tflite::ErrorReporter* errorReporter;
const tflite::Model* modelPointer;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Model class labels
const char* classLabels[3] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Set up error reporter
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  // Load the model
  modelPointer = tflite::GetModel(model);
  if (modelPointer->version() != TFLITE_SCHEMA_VERSION) {
    errorReporter->Report("Model version does not match Schema");
    return;
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter staticInterpreter(
    modelPointer, resolver, tensorArena, tensorArenaSize, errorReporter);
  interpreter = &staticInterpreter;

  // Allocate memory for tensors
  TfLiteStatus allocateStatus = interpreter->AllocateTensors();
  if (allocateStatus != kTfLiteOk) {
    errorReporter->Report("AllocateTensors() failed");
    return;
  }

  // Define model inputs
  input = interpreter->input(0);
  if (input->dims->size != 1 || input->dims->data[0] != 3) {
    errorReporter->Report("Input tensor has incorrect shape");
    return;
  }

  // Initialize the IMU sensor
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
}

void loop() {
  float ax, ay, az;
  if (!IMU.accelerationAvailable()) {
    Serial.println("Failed to read IMU data");
    return;
  }

  IMU.readAcceleration(ax, ay, az);

  // Copy the data from the sensor to model input buffer
  input->data.f[0] = ax;
  input->data.f[1] = ay;
  input->data.f[2] = az;

  // Invoke interpreter to perform inference
  if (interpreter->Invoke() != kTfLiteOk) {
    errorReporter->Report("Invoke failed");
    return;
  }

  // Process the output of model inference
  output = interpreter->output(0);
  uint8_t maxIndex = 0;
  for (int i = 1; i < 3; i++) {
    if (output->data.uint8[i] > output->data.uint8[maxIndex]) {
      maxIndex = i;
    }
  }

  // Output the result to serial
  Serial.println(classLabels[maxIndex]);

  // Pause before next reading
  delay(1000);
}
