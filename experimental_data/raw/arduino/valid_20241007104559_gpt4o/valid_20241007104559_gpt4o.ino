#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "model.h"

// Initialization Step: Include necessary libraries
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Declare Variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Define Tensor Arena
APDS9960 colorSensor(Wire, 2);  // Assuming interrupt pin is connected to pin 2
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  // Initialization Step: Set up Serial
  Serial.begin(9600);

  // Initialization Step: Set up the color sensor
  Wire.begin();
  if (!colorSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor");
    while (1);
  }

  // Load the Model
  tflite_model = ::tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version is not compatible");
    while (1);
  }

  // Resolve Operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate Memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Define Model Inputs
  input = interpreter->input(0);

  // Enable color sensor using public method
  if (!colorSensor.colorAvailable()) {  // Correct method to check color availability
    Serial.println("Error enabling color sensor");
    while (1);
  }
}

void loop() {
  // Preprocessing Step: Sensor Setup
  int red, green, blue, clear;
  if (!colorSensor.readColor(red, green, blue, clear)) {
    Serial.println("Error reading color values");
    return;
  }

  // Preprocessing Step: Optional Feature Extraction
  // Assuming RGB values are normalized between 0 and 1
  input->data.f[0] = static_cast<float>(red) / 65536;
  input->data.f[1] = static_cast<float>(green) / 65536;
  input->data.f[2] = static_cast<float>(blue) / 65536;

  // Inference Step: Invoke Interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Postprocessing Step: Process the output of model inference
  output = interpreter->output(0);
  int maxIndex = 0;
  float maxValue = output->data.f[0];
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > maxValue) {
      maxValue = output->data.f[i];
      maxIndex = i;
    }
  }

  // Output the classification result
  Serial.print("Object Classified as: ");
  Serial.println(classes[maxIndex]);
  delay(1000);  // Pause for a second before next reading
}
