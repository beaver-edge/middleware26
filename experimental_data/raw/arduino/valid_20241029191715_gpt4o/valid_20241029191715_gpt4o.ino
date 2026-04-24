#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h" // Ensure this is correctly included

// Initialization
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr; // Renamed to avoid conflict
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 1024;
uint8_t tensor_arena[kTensorArenaSize];

APDS9960 rgbSensor(Wire, /*interrupt_pin=*/2); // Provide necessary arguments
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  Serial.begin(9600);

  // Sensor Setup
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }

  // Load the Model
  tflite_model = tflite::GetModel(model); // Corrected to use 'model' from model.h
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Resolve Operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate Memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate memory for tensors!");
    while (1);
  }

  // Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Preprocessing
  int red, green, blue;
  if (!rgbSensor.readColor(red, green, blue)) {
    Serial.println("Failed to read color data!");
    return;
  }

  // Normalize and copy data to input tensor
  input->data.f[0] = static_cast<float>(red) / 65535.0;
  input->data.f[1] = static_cast<float>(green) / 65535.0;
  input->data.f[2] = static_cast<float>(blue) / 65535.0;

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke interpreter!");
    return;
  }

  // Postprocessing
  uint8_t class_idx = output->data.uint8[0];
  if (class_idx < 3) {
    Serial.print("Detected: ");
    Serial.println(classes[class_idx]);
  } else {
    Serial.println("Unknown object class");
  }

  delay(1000); // Delay for readability
}
