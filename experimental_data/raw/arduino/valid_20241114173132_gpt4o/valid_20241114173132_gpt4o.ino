#include <Arduino_APDS9960.h>
#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <Wire.h>

#include "model.h"

// Define the size of the tensor arena
constexpr int tensorArenaSize = 8 * 1024;
uint8_t tensorArena[tensorArenaSize];

// Declare critical variables for TensorFlow Lite
tflite::MicroErrorReporter errorReporter;
const tflite::Model* modelPointer;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Define classification labels
const int numClasses = 3;
const char* classes[numClasses] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

// Sensor setup
APDS9960 rgbSensor(Wire, -1);

void setup() {
  // Start serial communication
  Serial.begin(9600);

  // Initialize the RGB & Gesture sensor
  if (!rgbSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Load the model
  modelPointer = tflite::GetModel(model);
  if (modelPointer->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version does not match the supported version.");
    while (1);
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter staticInterpreter(
    modelPointer, resolver, tensorArena, tensorArenaSize, &errorReporter);
  interpreter = &staticInterpreter;

  // Allocate memory for the model's input and output tensors
  TfLiteStatus allocateStatus = interpreter->AllocateTensors();
  if (allocateStatus != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Define model inputs and outputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int red, green, blue;
  if (!rgbSensor.colorAvailable()) {
    return;
  }
  rgbSensor.readColor(red, green, blue);

  // Normalize to [0, 1] range as per the model's expectation
  input->data.f[0] = static_cast<float>(red) / 65535.0;
  input->data.f[1] = static_cast<float>(green) / 65535.0;
  input->data.f[2] = static_cast<float>(blue) / 65535.0;

  // Invoke the TensorFlow Lite interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error invoking the interpreter");
    return;
  }

  // Process the model's output
  uint8_t highestIndex = 0;
  for (uint8_t i = 1; i < numClasses; i++) {
    if (output->data.uint8[i] > output->data.uint8[highestIndex]) {
      highestIndex = i;
    }
  }

  // Output the detected class
  Serial.print("Detected: ");
  Serial.println(classes[highestIndex]);
  delay(1000);
}
