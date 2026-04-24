#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>

// Model data
#include "model.h"

// Constants
const int tensorArenaSize = 2 * 1024;
uint8_t tensorArena[tensorArenaSize];

// Variables for TensorFlow Lite
tflite::ErrorReporter* errorReporter = nullptr;
const tflite::Model* tfliteModel = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;

// APDS9960 sensor setup
APDS9960 rgbSensor(Wire, /* Interrupt Pin */ 2);

// Custom error reporter class
class CustomErrorReporter : public tflite::ErrorReporter {
public:
  int Report(const char* format, va_list args) override {
    vprintf(format, args);
    return 0;
  }
};

// Setup function
void setup() {
  Serial.begin(9600);

  // Initialize error reporter
  static CustomErrorReporter customErrorReporter;
  errorReporter = &customErrorReporter;

  // Load the model
  tfliteModel = tflite::GetModel(model);
  if (tfliteModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter staticInterpreter(
      tfliteModel, resolver, tensorArena, tensorArenaSize, errorReporter);
  interpreter = &staticInterpreter;

  // Allocate memory for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    errorReporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors
  inputTensor = interpreter->input(0);

  // Initialize the RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }
}

// Main loop
void loop() {
  // Read RGB values from the sensor
  int red, green, blue;
  if (!rgbSensor.colorAvailable() || !rgbSensor.readColor(red, green, blue)) {
    return;
  }

  // Normalize RGB values to range [0, 1]
  float r = red / 1024.0f;
  float g = green / 1024.0f;
  float b = blue / 1024.0f;

  // Copy normalized RGB values to input tensor
  inputTensor->data.f[0] = r;
  inputTensor->data.f[1] = g;
  inputTensor->data.f[2] = b;

  // Invoke the model
  if (interpreter->Invoke() != kTfLiteOk) {
    errorReporter->Report("Invoke failed.");
    return;
  }

  // Get the output from the model
  TfLiteTensor* outputTensor = interpreter->output(0);
  uint8_t classIndex = outputTensor->data.uint8[0];

  // Classes array
  const char* classes[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

  // Print the classified object to the serial port
  Serial.println(classes[classIndex]);

  // Delay before the next reading
  delay(1000);
}
