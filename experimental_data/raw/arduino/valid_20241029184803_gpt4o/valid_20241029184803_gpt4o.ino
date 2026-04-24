#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>
#include "model.h"

// Initialization of global variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int tensor_arena_size = 2048;
uint8_t tensor_arena[tensor_arena_size];

// APDS9960 color sensor object
APDS9960 apds(Wire, -1); // Assuming -1 for no interrupt pin

// Classification labels
const char* class_labels[] = {"Apple", "Banana", "Orange"};

class ArduinoErrorReporter : public tflite::ErrorReporter {
  int Report(const char* format, va_list args) override {
    vprintf(format, args);
    printf("\n");
    return 0;
  }
};

ArduinoErrorReporter arduino_error_reporter;

void setup() {
  // Setup Serial communication
  Serial.begin(9600);

  // Initialize the color sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Set up the TensorFlow Lite model
  error_reporter = &arduino_error_reporter;

  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while (1);
  }

  // Build the interpreter
  interpreter = new tflite::MicroInterpreter(
    tflite_model, resolver, tensor_arena, tensor_arena_size, error_reporter);

  // Allocate memory for the model's tensors
  interpreter->AllocateTensors();

  // Get pointers to the input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read color data from the APDS9960 sensor
  int red, green, blue;
  if (!apds.readColor(red, green, blue)) {
    Serial.println("Error reading color data!");
    return;
  }

  // Normalize the RGB sensor data and copy it to the input tensor
  input->data.f[0] = static_cast<float>(red) / 65535.0;
  input->data.f[1] = static_cast<float>(green) / 65535.0;
  input->data.f[2] = static_cast<float>(blue) / 65535.0;

  // Run the model inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error invoking the interpreter!");
    return;
  }

  // Process the model output and print the classified object as an emoji
  int max_index = 0;
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > output->data.f[max_index]) {
      max_index = i;
    }
  }

  // Unicode emoji output based on classification
  Serial.print("Detected Object: ");
  switch (max_index) {
    case 0: Serial.println("🍎"); break; // Apple
    case 1: Serial.println("🍌"); break; // Banana
    case 2: Serial.println("🍊"); break; // Orange
    default: Serial.println("Unknown"); break;
  }

  delay(1000); // Delay for a second before next reading
}
