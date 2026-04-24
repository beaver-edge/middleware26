#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Initialization step: Declare necessary variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 4096;
uint8_t tensor_arena[kTensorArenaSize];

// Preprocessing step: Set up the sensor
APDS9960 apds(Wire, -1);  // Ensure to pass the correct parameters

// Classification labels
const char* labels[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  // Set up serial communication
  Serial.begin(9600);

  // Initialize error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    return;
  }

  // Define model inputs
  input = interpreter->input(0);

  // Set up the sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }
}

void loop() {
  // Read RGB values from the sensor
  int red, green, blue;
  if (!apds.colorAvailable()) {
    Serial.println("Error reading RGB values!");
    return;
  }
  
  apds.readColor(red, green, blue);

  // Normalize RGB values to [0, 1]
  input->data.f[0] = red / 255.0;
  input->data.f[1] = green / 255.0;
  input->data.f[2] = blue / 255.0;

  // Inference step: Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error invoking the interpreter!");
    return;
  }

  // Postprocessing step: Process the output of model inference
  output = interpreter->output(0);
  uint8_t class_index = output->data.uint8[0];
  Serial.print("Detected object: ");
  Serial.println(labels[class_index]);

  // Add a delay to avoid flooding the serial port
  delay(1000);
}
