#include <Arduino_APDS9960.h>
#include <Arduino_HTS221.h>
#include <TensorFlowLite.h>
#include "model.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Initialization - Declare Variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define Tensor Arena
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Sensor instance
APDS9960 apds(Wire, 2);  // Assuming interrupt pin is 2, adjust as necessary

// Initialization
void setup() {
  // Serial setup
  Serial.begin(9600);
  while (!Serial);

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version does not match Schema");
    while (1);
  }

  // Resolve Operators
  tflite::AllOpsResolver resolver;

  // Instantiate the Interpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate Memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Sensor setup
  if (!apds.begin()) {
    Serial.println("Error initializing APDS-9960 sensor.");
    while (1);
  }
}

// Main Loop
void loop() {
  int r, g, b, c;
  
  // Read color data
  if (!apds.colorAvailable()) {
    return;
  }
  apds.readColor(r, g, b, c);

  // Preprocessing - Normalize sensor data
  input->data.f[0] = r / 255.0;
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Inference - Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed.");
    return;
  }

  // Postprocessing - Interpret results and output
  uint8_t class_index = output->data.uint8[0];
  switch (class_index) {
    case 0:
      Serial.println("🍎 Apple");
      break;
    case 1:
      Serial.println("🍌 Banana");
      break;
    case 2:
      Serial.println("🍊 Orange");
      break;
    default:
      Serial.println("Unknown object");
      break;
  }

  delay(1000);  // Delay for readability
}
