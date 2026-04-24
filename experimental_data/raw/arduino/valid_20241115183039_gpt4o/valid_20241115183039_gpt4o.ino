#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Declare Variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define Tensor Arena
constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// Load the Model
#include "model.h"  // Ensure model.h contains the header for model data
extern const unsigned char model[]; 
extern const int model_length;

// Initialize the RGB Sensor
APDS9960 rgbSensor(Wire, /*intPin*/ -1); // Provide appropriate Wire and interrupt pin

// Define classes for output
const char* class_labels[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

void setup() {
  // Start Serial communication
  Serial.begin(9600);

  // Initialize RGB Sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS-9960 sensor!");
    while (1);
  }

  // Load Model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while (1);
  }

  // Resolve Operators
  static tflite::AllOpsResolver resolver;

  // Instantiate Interpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate Memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Define Model Inputs
  input = interpreter->input(0);
}

void loop() {
  int r, g, b;

  // Sensor Setup and Data Collection
  if (rgbSensor.colorAvailable()) {
    rgbSensor.readColor(r, g, b);

    // Preprocessing: Normalize color values
    float red = static_cast<float>(r) / 65535.0;
    float green = static_cast<float>(g) / 65535.0;
    float blue = static_cast<float>(b) / 65535.0;

    // Copy data to input tensor
    input->data.f[0] = red;
    input->data.f[1] = green;
    input->data.f[2] = blue;

    // Invoke interpreter
    if (interpreter->Invoke() != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", 0);
      return;
    }

    // Process the output of model inference
    output = interpreter->output(0);
    uint8_t classificationIndex = output->data.uint8[0];

    // Output classification result
    Serial.println(class_labels[classificationIndex]);
  }

  delay(500); // Delay for readability
}
