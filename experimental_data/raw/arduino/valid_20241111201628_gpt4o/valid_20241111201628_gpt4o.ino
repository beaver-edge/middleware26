#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "model.h"

#define TENSOR_ARENA_SIZE 2 * 1024

// Critical variables for TensorFlow Lite Micro
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// Tensor arena buffer
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Output labels
const char* output_labels[3] = {"🍏", "🍌", "🍊"};

void setup() {
  // Begin serial communication
  Serial.begin(9600);
  
  // Initialize the color sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }
  
  // Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  tflite_model = tflite::GetModel(model);
  
  // Set up operator resolver
  static tflite::AllOpsResolver micro_op_resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory for the model's input and output tensors
  interpreter->AllocateTensors();

  // Define model input
  input = interpreter->input(0);
}

void loop() {
  // Variables to store color data
  int r, g, b;
  
  // Get the color data from sensor
  if (!APDS.colorAvailable()) {
    delay(10);
    return;
  }
  
  APDS.readColor(r, g, b);
  
  // Convert the RGB values to float and normalize
  input->data.f[0] = r / 255.0;
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed!");
    return;
  }

  // Get the output probabilities
  TfLiteTensor* output = interpreter->output(0);
  
  // Find the highest scoring label
  int maxIndex = 0;
  float maxValue = output->data.f[0];
  
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > maxValue) {
      maxValue = output->data.f[i];
      maxIndex = i;
    }
  }
  
  // Print the classified object
  Serial.println(output_labels[maxIndex]);
  
  // Delay for 1 second
  delay(1000);
}
