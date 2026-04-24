#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Constants
const int kTensorArenaSize = 1024;
const char* classes[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

// TensorFlow Lite model variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model_ptr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
uint8_t tensor_arena[kTensorArenaSize];

// Color sensor variables
int r, g, b;

void setup() {
  // Initialize serial for output
  Serial.begin(9600);
  while (!Serial);

  // Initialize the color sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Initialize error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  model_ptr = tflite::GetModel(model);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.",
                           model_ptr->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model_ptr, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Obtain pointers to the model’s input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Set up the RGB sensor
  APDS.setGestureSensitivity(80);  // Use an alternative available public method to interact with the sensor
}

void loop() {
  // Read sensor data
  if (!APDS.colorAvailable()) {
    return;
  }
  APDS.readColor(r, g, b);

  // Normalize RGB values to [0, 1] and set input tensor values
  input->data.f[0] = r / 255.0;
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Process the output and print the result
  int classIndex = output->data.uint8[0];
  Serial.println(classes[classIndex]);

  // Wait for a second before taking a new reading
  delay(1000);
}
