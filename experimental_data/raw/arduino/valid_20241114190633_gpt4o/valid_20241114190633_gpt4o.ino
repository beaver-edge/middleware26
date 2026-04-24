#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/kernels/micro_ops.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Initialization of libraries and variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *tf_model;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;

constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Classification classes
const char* kClasses[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

// Sensor initialization with Wire object
APDS9960 rgbSensor(Wire, 0); // Pass Wire object and an invalid value for intPin (0) as it's not used here

void setup() {
  // Setup serial communication for debugging
  Serial.begin(9600);
  while (!Serial);

  // Initialize the RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Error initializing sensor!");
    while (1);
  }
  Serial.println("RGB sensor initialized.");

  // Load the TensorFlow model from the header file
  tf_model = tflite::GetModel(model);
  if (tf_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version mismatch.");
    return;
  }

  // Op resolver
  static tflite::AllOpsResolver resolver;

  // Interpreter instantiation
  static tflite::MicroInterpreter static_interpreter(
    tf_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensor space
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Input and output tensor pointers setup
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Get RGB readings
  int r, g, b;
  if (!rgbSensor.readColor(r, g, b)) {
    Serial.println("Error reading color");
    return;
  }

  // Preprocessing: Normalize the RGB values
  float red_norm = r / 255.0;
  float green_norm = g / 255.0;
  float blue_norm = b / 255.0;

  // Fill input tensor buffer
  input->data.f[0] = red_norm;
  input->data.f[1] = green_norm;
  input->data.f[2] = blue_norm;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Postprocessing: Get the output and classify
  int classIdx = 0;
  for (int i = 1; i < 3; ++i) {
    if (output->data.uint8[i] > output->data.uint8[classIdx]) {
      classIdx = i;
    }
  }

  // Output classification result
  Serial.println(kClasses[classIdx]);

  // Delay to avoid spamming
  delay(1000);
}
