#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>
#include <Wire.h>

// Model and TensorFlow Lite Micro specifics
#include "model.h"  // Include the TFLite model file

constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Assuming the APDS9960 library requires TwoWire and an interrupt pin
const int intPin = 2; // Example interrupt pin, check your setup
APDS9960 apds(Wire, intPin);

// Object classification classes
const char* classes[] = {"🍏", "🍌", "🍊"};  // Apple, Banana, Orange

void setup() {
  Serial.begin(9600);
  Wire.begin();

  // Initialize sensor
  if (!apds.begin()) {
    Serial.println("Failed to initialize color sensor!");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Set up the resolver and interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (1);
  }

  // Get model input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Ensure input size matches model expectations
  if (input->dims->size != 2 || input->dims->data[0] != 1 || input->dims->data[1] != 3) {
    Serial.println("Unexpected input tensor shape!");
    while (1);
  }
}

void loop() {
  // Read RGB data from the sensor
  int r, g, b, c;
  if (!apds.colorAvailable()) {
    return;
  }
  apds.readColor(r, g, b, c);

  // Normalize the RGB values
  float red = r / 65535.0;
  float green = g / 65535.0;
  float blue = b / 65535.0;

  // Copy normalized data to input tensor
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Process the output from the model
  uint8_t class_index = output->data.uint8[0];
  if (class_index < 3) {
    Serial.print("Detected: ");
    Serial.println(classes[class_index]);
  } else {
    Serial.println("Unknown class index!");
  }

  delay(1000);  // Delay for demonstration purposes
}
