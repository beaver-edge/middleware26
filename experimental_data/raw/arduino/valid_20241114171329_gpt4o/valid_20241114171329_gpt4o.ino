#include <Arduino.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// Initialization: Include Necessary Libraries
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model_instance = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define Tensor Arena
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Neural Network Model Classes
const char* kClasses[] = {"Apple", "Banana", "Orange"};

// Placeholder for Sensor Data
float norm_r = 0.0f;
float norm_g = 0.0f;
float norm_b = 0.0f;

void setup() {
  // Start Serial Communication
  Serial.begin(9600);

  // Initialization: Load the Model
  model_instance = tflite::GetModel(model);
  if (model_instance->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version does not match!");
    return;
  }

  // Initialization: Resolving Operators
  static tflite::MicroMutableOpResolver<2> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  // Initialization: Instantiate the Interpreter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  static tflite::MicroInterpreter static_interpreter(
      model_instance, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Initialization: Allocate Memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Initialization: Define Model Inputs
  input = interpreter->input(0);
}

void loop() {
  // Simulate Sensor Data (as we can't use Adafruit_TCS34725 without library)
  norm_r = random(0, 100) / 100.0f;
  norm_g = random(0, 100) / 100.0f;
  norm_b = random(0, 100) / 100.0f;

  // Inference: Data Copy
  input->data.f[0] = norm_r;
  input->data.f[1] = norm_g;
  input->data.f[2] = norm_b;

  // Inference: Invoke Interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Postprocessing: Process the Output of Model Inference
  output = interpreter->output(0);
  uint8_t max_index = 0;
  float max_value = 0.0;
  for (int i = 0; i < 3; i++) {
    float value = output->data.f[i];
    if (value > max_value) {
      max_value = value;
      max_index = i;
    }
  }

  // Output the result via Serial using Unicode Emojis
  const char* emoji;
  switch (max_index) {
    case 0:
      emoji = "\U0001F34E";  // Apple Emoji
      break;
    case 1:
      emoji = "\U0001F34C";  // Banana Emoji
      break;
    case 2:
      emoji = "\U0001F34A";  // Orange Emoji
      break;
    default:
      emoji = "?";
      break;
  }

  Serial.print("Detected: ");
  Serial.println(emoji);

  delay(2000); // Delay between readings
}
