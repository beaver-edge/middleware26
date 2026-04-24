#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"  // Ensure model.h contains the correct model data

// Declare variables for TensorFlow Lite Micro
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena for memory allocation
constexpr int kTensorArenaSize = 1024;
uint8_t tensor_arena[kTensorArenaSize];

// RGB Sensor
APDS9960 rgbSensor(Wire, 2); // Assuming pin 2 is used for interrupt

// Classification map
const char* classNames[] = {"Apple", "Banana", "Orange"};

class ArduinoErrorReporter : public tflite::ErrorReporter {
  int Report(const char* format, va_list args) override {
    vprintf(format, args);
    return 0;
  }
};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Set up the RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize the RGB sensor!");
    while (1);
  }

  // Load the model and allocate memory
  tflite_model = tflite::GetModel(model); // Correctly refer to the model data
  static tflite::AllOpsResolver resolver;

  static ArduinoErrorReporter error_reporter_impl;
  error_reporter = &error_reporter_impl;

  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read RGB data from the sensor
  int red, green, blue;
  if (!rgbSensor.readColor(red, green, blue)) {
    Serial.println("Failed to read RGB values!");
    return;
  }

  // Normalize and copy data to the input tensor
  input->data.f[0] = red / 65535.0f;
  input->data.f[1] = green / 65535.0f;
  input->data.f[2] = blue / 65535.0f;

  // Invoke the interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke the interpreter!");
    return;
  }

  // Process the output to determine the class
  uint8_t maxIndex = 0;
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.f[i] > output->data.f[maxIndex]) {
      maxIndex = i;
    }
  }

  // Output the class
  Serial.println(classNames[maxIndex]);

  delay(1000); // Delay for a while before the next reading
}
