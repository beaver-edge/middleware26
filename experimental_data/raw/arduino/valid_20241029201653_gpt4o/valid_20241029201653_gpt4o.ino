#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// TensorFlow Lite model includes
#include "tensorflow/lite/micro/all_ops_resolver.h"

// Constants
const char* class_labels[] = {"Apple", "Banana", "Orange"};
const int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Global variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Sensor object
APDS9960 rgbSensor(Wire, -1); // Assuming no interrupt pin is used

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize the error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.", tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Resolve needed operators
  static tflite::AllOpsResolver resolver;

  // Build the interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Setup input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize the RGB sensor
  Wire.begin();
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }
  Serial.println("APDS9960 sensor initialized.");
}

void loop() {
  int red, green, blue, proximity;

  // Read data from the RGB sensor
  if (!rgbSensor.readColor(red, green, blue, proximity)) {
    Serial.println("Failed to read color data!");
    return;
  }

  // Normalize the RGB values and copy to the input tensor
  input->data.f[0] = static_cast<float>(red) / 255.0;
  input->data.f[1] = static_cast<float>(green) / 255.0;
  input->data.f[2] = static_cast<float>(blue) / 255.0;

  // Run the model
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Get the model's output and determine the class
  uint8_t predicted_class = output->data.uint8[0];
  Serial.print("Predicted Class: ");
  Serial.println(class_labels[predicted_class]);
  delay(1000);
}
