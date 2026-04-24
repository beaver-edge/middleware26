#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "./model.h"

// Declare variables for TensorFlow Lite
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* trained_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define the tensor arena size
constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// Define the classification labels
const char* class_labels[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  // Set up the serial communication
  Serial.begin(9600);
  
  // Initialize the error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model from the flatbuffer data
  trained_model = tflite::GetModel(model);
  if (trained_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal "
                           "to supported version %d.",
                           trained_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Set up the resolver with the necessary built-in operators
  static tflite::AllOpsResolver micro_op_resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      trained_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor arena for the model
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize the RGB sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
  }
}

void loop() {
  // Read the sensor data
  int red, green, blue;
  if (APDS.colorAvailable()) {
    APDS.readColor(red, green, blue);

    // Normalize the RGB values to the dataset scale [0, 1]
    input->data.f[0] = static_cast<float>(red) / 255.0;
    input->data.f[1] = static_cast<float>(green) / 255.0;
    input->data.f[2] = static_cast<float>(blue) / 255.0;

    // Invoke the TensorFlow Lite model
    if (interpreter->Invoke() != kTfLiteOk) {
      error_reporter->Report("Invoke failed");
      return;
    }

    // Process the output
    int max_index = 0;
    for (int i = 0; i < 3; i++) {
      if (output->data.uint8[i] > output->data.uint8[max_index]) {
        max_index = i;
      }
    }

    // Output the predicted class label with Unicode emoji
    Serial.println(class_labels[max_index]);
  }

  // Wait a bit before the next reading
  delay(500);
}
