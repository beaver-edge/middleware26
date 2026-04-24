#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// Declare variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model_pointer = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

APDS9960 colorSensor(Wire, -1);  // Initialize with Wire and no interrupt pin
int red, green, blue;

void setup() {
  // Initialize Serial communication
  Serial.begin(9600);

  // Set up error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  model_pointer = tflite::GetModel(model);
  if (model_pointer->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal "
                                          "to supported version %d.",
                         model_pointer->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Set up the op resolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model_pointer, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Define model input
  input = interpreter->input(0);

  // Set up color sensor
  if (!colorSensor.begin()) {
    Serial.println("Error initializing color sensor");
    while (1);
  }
}

void loop() {
  // Collect RGB data from the sensor
  if (!colorSensor.readColor(red, green, blue)) {
    Serial.println("Error reading color");
    return;
  }

  // Normalize and copy data to the model's input buffer
  input->data.f[0] = red / 255.0;
  input->data.f[1] = green / 255.0;
  input->data.f[2] = blue / 255.0;

  // Invoke the interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // Process the output
  output = interpreter->output(0);
  int max_index = 0;
  for (int i = 1; i < 3; ++i) {
    if (output->data.uint8[i] > output->data.uint8[max_index]) {
      max_index = i;
    }
  }

  // Output the classification result
  switch (max_index) {
    case 0:
      Serial.println("Classified: 🍎 Apple");
      break;
    case 1:
      Serial.println("Classified: 🍌 Banana");
      break;
    case 2:
      Serial.println("Classified: 🍊 Orange");
      break;
  }

  delay(1000); // Wait for 1 second before next reading
}
