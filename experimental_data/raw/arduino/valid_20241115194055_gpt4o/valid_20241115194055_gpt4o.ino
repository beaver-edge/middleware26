#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Declare necessary variables for TinyML
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model_pointer = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int tensor_arena_size = 2 * 1024;
byte tensor_arena[tensor_arena_size];

// Classification labels
const char* classification_labels[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

// Setup the color sensor
APDS9960 color_sensor(Wire, -1); // Assuming -1 if no interrupt pin is used

// Initialization
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  model_pointer = tflite::GetModel(model);
  if (model_pointer->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, 
                         "Model schema version is not equal to TFLITE_SCHEMA_VERSION.");
    return;
  }

  // Initialize the resolver and interpreter
  tflite::AllOpsResolver resolver;
  interpreter = new tflite::MicroInterpreter(
      model_pointer, resolver, tensor_arena, tensor_arena_size, error_reporter);

  // Allocate memory
  interpreter->AllocateTensors();

  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize color sensor
  if (!color_sensor.begin()) {
    Serial.println("Failed to initialize color sensor!");
    while (1);
  }
}

// Main loop for inference
void loop() {
  // Preprocessing: Read color data
  int red, green, blue;
  if (!color_sensor.colorAvailable()) {
    delay(100);  // Wait for new data
    return;
  }
  color_sensor.readColor(red, green, blue);

  // Normalize color data
  input->data.f[0] = red / 1024.0;
  input->data.f[1] = green / 1024.0;
  input->data.f[2] = blue / 1024.0;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error invoking TFLite model!");
    return;
  }

  // Get the output and identify the class
  uint8_t highest_index = 0;
  uint8_t* results = output->data.uint8;

  for (int i = 1; i < 3; i++) {
    if (results[i] > results[highest_index]) {
      highest_index = i;
    }
  }

  // Output result
  Serial.println(classification_labels[highest_index]);

  delay(1000);  // Delay before next inference
}
