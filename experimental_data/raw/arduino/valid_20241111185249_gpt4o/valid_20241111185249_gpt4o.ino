#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// TensorFlow Lite Micro variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Include model data
#include "model.h"

// APDS-9960 Color Sensor Setup
TwoWire& i2c = Wire;  // Use the default Wire object
int interruptPin = -1; // No interrupt pin required for color sensor usage
APDS9960 colorSensor(i2c, interruptPin);

// Classes for classification
const char* CLASSES[3] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

void setup() {
  // Start Serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize color sensor
  if (!colorSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version is not compatible.");
    while (1);
  }

  // Define the operator resolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed.");
    while (1);
  }

  // Obtain input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Output initialization success
  Serial.println("Initialization done.");
}

void loop() {
  int r, g, b;
  
  // Check if color available and read RGB values
  if (!colorSensor.colorAvailable()) {
    delay(100);
    return;
  }
  
  colorSensor.readColor(r, g, b);

  // Normalize RGB values to float and copy to TensorFlow Lite input
  input->data.f[0] = r / 255.0f;
  input->data.f[1] = g / 255.0f;
  input->data.f[2] = b / 255.0f;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed.");
    return;
  }

  // Process output and print classification result to Serial
  uint8_t max_index = 0;
  for (int i = 1; i < 3; i++) {
    if (output->data.uint8[i] > output->data.uint8[max_index]) {
      max_index = i;
    }
  }

  Serial.print("Object detected: ");
  Serial.println(CLASSES[max_index]);

  // Add a delay before next reading
  delay(1000);
}
