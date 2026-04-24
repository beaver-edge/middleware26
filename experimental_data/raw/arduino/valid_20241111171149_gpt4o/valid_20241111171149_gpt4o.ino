#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Declare critical variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;

// Reserve memory for the tensor arena
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Define classes for classification
const char* CLASSES[] = { "Apple 🍏", "Banana 🍌", "Orange 🍊" };

// Include the trained model
#include "model.h"

// APDS9960 Color sensor setup
APDS9960 colorSensor(Wire, 2); // Assuming pin 2 for interrupt

void setup() {
  // Set up serial communication
  Serial.begin(9600);
  
  // Set up the error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version not supported");
    return;
  }

  // Set up the resolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Define the input tensor
  input_tensor = interpreter->input(0);

  // Initialize the color sensor
  if (!colorSensor.begin()) {
    Serial.println("Failed to initialize the color sensor!");
    while(1);
  }
}

void loop() {
  int r, g, b;
  
  // Read the RGB values
  if (!colorSensor.colorAvailable()) {
    return;
  }
  colorSensor.readColor(r, g, b);
  
  // Normalize and prepare the sensor data to feed into the model
  input_tensor->data.f[0] = r / 255.0;
  input_tensor->data.f[1] = g / 255.0;
  input_tensor->data.f[2] = b / 255.0;
  
  // Invoke the interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Get the output of model inference
  TfLiteTensor* output = interpreter->output(0);
  int classIndex = output->data.uint8[0];
  
  // Output the classified object
  Serial.println(CLASSES[classIndex]);

  // Delay for a while before the next reading
  delay(1000);
}
