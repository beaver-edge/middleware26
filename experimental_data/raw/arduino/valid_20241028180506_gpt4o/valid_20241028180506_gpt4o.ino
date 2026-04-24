#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Constants
const int tensorArenaSize = 2 * 1024;
const char* classLabels[] = {"🍎 (Apple)", "🍌 (Banana)", "🍊 (Orange)"};

// Global variables
tflite::MicroErrorReporter microErrorReporter;
tflite::ErrorReporter* errorReporter = &microErrorReporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
byte tensorArena[tensorArenaSize];
APDS9960 rgbSensor(Wire, 2); // Assuming pin 2 for interrupt, modify as needed

// Setup function
void setup() {
  Serial.begin(9600);
  Wire.begin();

  // Initialize RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Load model
  const unsigned char* modelData = nullptr; // Replace with actual model data
  model = tflite::GetModel(modelData);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    errorReporter->Report("Model version does not match Schema");
    while (1);
  }

  // Set up the resolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(model, resolver, tensorArena, tensorArenaSize, errorReporter);

  // Allocate memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    errorReporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Define model input
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// Main loop
void loop() {
  int red, green, blue, clear;

  // Read RGB values from the sensor
  if (rgbSensor.colorAvailable()) {
    rgbSensor.readColor(red, green, blue, clear);

    // Normalize RGB values and copy to input tensor
    input->data.f[0] = red / 65535.0f;
    input->data.f[1] = green / 65535.0f;
    input->data.f[2] = blue / 65535.0f;

    // Invoke the interpreter
    if (interpreter->Invoke() != kTfLiteOk) {
      errorReporter->Report("Invoke failed");
      return;
    }

    // Process the output
    int8_t predictedIndex = output->data.uint8[0]; // Assuming output is of uint8 type
    Serial.println(classLabels[predictedIndex]);
  }

  delay(1000);  // Delay to avoid flooding the serial output
}
