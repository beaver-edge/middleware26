#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model.h"

// TensorFlow Lite Micro related variables
tflite::MicroErrorReporter error_reporter;
const tflite::Model* tflite_model = nullptr;  // Renamed to avoid conflict
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 2 * 1024;
byte tensor_arena[kTensorArenaSize];

// APDS9960 sensor instance
APDS9960 apds(Wire, -1);  // Provide the correct parameters for initialization

// Object classes
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  Serial.begin(9600);
  Wire.begin();

  // Initialize the sensor
  if (!apds.begin()) {
    Serial.println("Failed to initialize APDS9960!");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);  // Use the correct model variable
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version does not match!");
    while (1);
  }

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

  // Allocate memory for the model's tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    while (1);
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int red, green, blue, proximity;  // Change to int to match function signature

  // Read color data from the sensor
  if (!apds.readColor(red, green, blue, proximity)) {
    Serial.println("Error reading color!");
    return;
  }

  // Normalize the RGB values
  float norm_red = red / 1024.0;
  float norm_green = green / 1024.0;
  float norm_blue = blue / 1024.0;

  // Copy the normalized values to the input tensor
  input->data.f[0] = norm_red;
  input->data.f[1] = norm_green;
  input->data.f[2] = norm_blue;

  // Perform inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error invoking the interpreter!");
    return;
  }

  // Process the output
  uint8_t max_index = 0;
  float max_value = output->data.f[0];
  for (uint8_t i = 1; i < 3; ++i) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }

  // Output the classification result
  Serial.print("Detected object: ");
  Serial.println(classes[max_index]);
  delay(1000);
}
