#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Include the model header
#include "model.h"

// TensorFlow Lite objects
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;
const tflite::Model *loaded_model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// Tensor arena size
constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// Classification labels
const char *kClasses[] = {"Apple", "Banana", "Orange"};

// RGB Sensor setup
APDS9960 rgbSensor(Wire, -1);

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  // Setup APDS-9960 sensor
  if (!rgbSensor.begin()) {
    Serial.println("Error initializing APDS-9960 sensor.");
    while (1);
  }

  // Load model from flash
  loaded_model = tflite::GetModel(model);
  if (loaded_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema.");
    while (1);
  }

  // Resolve all operators
  static tflite::AllOpsResolver resolver;

  // Instantiate interpreter
  static tflite::MicroInterpreter static_interpreter(
    loaded_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Check if color data is available
  if (!rgbSensor.colorAvailable()) {
    return;
  }

  // Read RGB values from the sensor
  int red, green, blue;
  rgbSensor.readColor(red, green, blue);

  // Normalize RGB values to match model input expectations (0-1 range)
  input->data.f[0] = red / 255.0;
  input->data.f[1] = green / 255.0;
  input->data.f[2] = blue / 255.0;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Find the object class with highest score
  uint8_t max_index = 0;
  uint8_t max_value = output->data.uint8[0];
  for (uint8_t i = 1; i < output->dims->data[0]; i++) {
    if (output->data.uint8[i] > max_value) {
      max_value = output->data.uint8[i];
      max_index = i;
    }
  }

  // Output the result to the serial
  Serial.print("Detected: ");
  Serial.println(kClasses[max_index]);
}
