#include <Wire.h>
#include <SPI.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"  // Include the correct header for MicroErrorReporter
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "Arduino_APDS9960.h"
#include "./model.h"

// Error reporting
tflite::ErrorReporter* error_reporter = nullptr;
// The model to be used
const tflite::Model* tensor_model = nullptr;
// TensorFlowLite operations resolver
tflite::AllOpsResolver resolver;
// TensorFlowLite Interpreter
tflite::MicroInterpreter* interpreter = nullptr;

// Input and output from the model
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow's allocations.
constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// Color sensor
APDS9960 apds(Wire, -1); // Assuming default Wire instance and not using interrupt pin

// Object classes
const char* object_classes[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  // Start the serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize the color sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS-9960 sensor!");
    while (1);
  }
  Serial.println("APDS-9960 found!");

  // Set up error reporting
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  tensor_model = tflite::GetModel(model);
  if (tensor_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.",
                           tensor_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(tensor_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read color from the sensor
  int r, g, b;
  if (!apds.colorAvailable()) {
    return;
  }
  apds.readColor(r, g, b);

  // Normalize and prepare input data
  input->data.f[0] = float(r) / 1024.0;
  input->data.f[1] = float(g) / 1024.0;
  input->data.f[2] = float(b) / 1024.0;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on the input data.");
    return;
  }

  // Process the inference results
  uint8_t class_index = output->data.uint8[0];
  Serial.print("Detected class: ");
  Serial.println(object_classes[class_index]);

  // Wait for a second
  delay(1000);
}
