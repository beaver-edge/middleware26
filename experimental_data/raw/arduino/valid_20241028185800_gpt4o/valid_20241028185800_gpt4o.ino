#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Model data
#include "model.h"

// Constants
#define TENSOR_ARENA_SIZE 2 * 1024

// Global variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Sensor instance
APDS9960 rgbSensor(Wire, 2); // Assuming intPin is 2, adjust as needed

// Setup function
void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Initialize the APDS9960 sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }
  Serial.println("APDS9960 sensor initialized.");

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.", tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get pointers to the input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

// Main loop
void loop() {
  int red, green, blue, ambient;
  
  // Read RGB values from the sensor
  if (!rgbSensor.colorAvailable()) {
    delay(50);
    return;
  }
  
  rgbSensor.readColor(red, green, blue, ambient);

  // Normalize and copy sensor data to input tensor
  input->data.f[0] = static_cast<float>(red) / 65535.0f;
  input->data.f[1] = static_cast<float>(green) / 65535.0f;
  input->data.f[2] = static_cast<float>(blue) / 65535.0f;

  // Perform inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Process the output
  uint8_t predicted_class = output->data.uint8[0];
  
  // Output the result
  switch (predicted_class) {
    case 0:
      Serial.println("🍎");  // Apple
      break;
    case 1:
      Serial.println("🍌");  // Banana
      break;
    case 2:
      Serial.println("🍊");  // Orange
      break;
    default:
      Serial.println("Unknown");
      break;
  }

  // Add a small delay to avoid flooding the serial output
  delay(500);
}
