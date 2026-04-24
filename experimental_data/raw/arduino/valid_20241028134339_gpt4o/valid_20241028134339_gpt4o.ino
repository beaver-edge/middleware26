#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"  // Ensure this is in the same directory

// Declare variables
tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define Tensor Arena
constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// Initialize the sensor with default Wire and interrupt pin
APDS9960 rgbSensor(Wire, -1); // Using default I2C Wire and no interrupt pin

// Classes
const char* class_names[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

void setup() {
  Serial.begin(9600);

  // Sensor setup
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960!");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
  
  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  // Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Preprocessing: Read data from sensor
  int red, green, blue, ambient;
  if (!rgbSensor.readColor(red, green, blue, ambient)) {
    Serial.println("Error reading color values!");
    return;
  }

  // Normalize the sensor values
  float red_normalized = red / 255.0;
  float green_normalized = green / 255.0;
  float blue_normalized = blue / 255.0;

  // Data copy: Fill input tensor
  input->data.f[0] = red_normalized;
  input->data.f[1] = green_normalized;
  input->data.f[2] = blue_normalized;

  // Inference: Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error invoking model!");
    return;
  }

  // Postprocessing: Process the output
  uint8_t class_index = output->data.uint8[0];
  if (class_index < sizeof(class_names) / sizeof(class_names[0])) {
    Serial.print("Detected: ");
    Serial.println(class_names[class_index]);
  } else {
    Serial.println("Unknown class!");
  }

  delay(1000);  // Delay between inferences
}
