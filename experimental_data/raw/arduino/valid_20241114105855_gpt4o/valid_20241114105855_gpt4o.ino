#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

// Define constants
#define TENSOR_ARENA_SIZE 2048

// Declare variables
tflite::ErrorReporter* errorReporter;
const tflite::Model* tfModel;
tflite::MicroInterpreter* interpreter;
tflite::AllOpsResolver resolver;
TfLiteTensor* input;
TfLiteTensor* output;
uint8_t tensorArena[TENSOR_ARENA_SIZE];

// Possible classes
const char* classes[] = { "🍎 Apple", "🍌 Banana", "🍊 Orange" };

// Initialize the APDS9960 sensor
APDS9960 rgbSensor(Wire, -1);

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Initialize the error reporter
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  // Load the TFLite model
  tfModel = tflite::GetModel(model);
  if (tfModel == nullptr) {
    errorReporter->Report("Failed to load model.");
    return;
  }

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(
      tfModel, resolver, tensorArena, TENSOR_ARENA_SIZE, errorReporter);

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    errorReporter->Report("AllocateTensors() failed");
    return;
  }

  // Set up model input
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize the RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }
}

void loop() {
  // Read RGB values from the sensor
  int r, g, b;
  if (!rgbSensor.colorAvailable() || !rgbSensor.readColor(r, g, b)) {
    Serial.println("Failed to read color data!");
    return;
  }

  // Normalize and copy to input
  input->data.f[0] = r / 255.0;
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Perform inference
  if (interpreter->Invoke() != kTfLiteOk) {
    errorReporter->Report("Failed to run inference");
    return;
  }

  // Find the class with the highest score
  uint8_t predictedClass = 0;
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.uint8[i] > output->data.uint8[predictedClass]) {
      predictedClass = i;
    }
  }

  // Output the result
  Serial.print("Predicted class: ");
  Serial.println(classes[predictedClass]);

  // Sleep for a while
  delay(2000);
}
