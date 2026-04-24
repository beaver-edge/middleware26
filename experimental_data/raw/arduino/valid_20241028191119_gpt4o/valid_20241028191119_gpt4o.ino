#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// TensorFlow Lite Model
#include "model.h"

// Constants
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// TensorFlow Lite Global Variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// APDS-9960 Sensor
APDS9960 rgbSensor(Wire, -1); // Provide Wire and interrupt pin

// Classification labels
const char* class_labels[] = {"Apple", "Banana", "Orange"};

void setup() {
  // Initialize Serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize I2C communication
  Wire.begin();

  // Initialize the RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS-9960 sensor!");
    while (1);
  }
  
  // Load the TensorFlow Lite model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version does not match!");
    while (1);
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    while (1);
  }

  // Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int r, g, b, c;

  // Read RGB values from the sensor
  if (!rgbSensor.readColor(r, g, b, c)) {
    Serial.println("Failed to read RGB values!");
    return;
  }

  // Preprocess sensor data
  float red = r / 1024.0;
  float green = g / 1024.0;
  float blue = b / 1024.0;
  
  // Copy data to input tensor
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // Invoke the interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke interpreter!");
    return;
  }

  // Post-process the output
  uint8_t class_index = output->data.uint8[0];
  if (class_index < sizeof(class_labels) / sizeof(class_labels[0])) {
    Serial.println(class_labels[class_index]);
  } else {
    Serial.println("Unknown class!");
  }

  // Wait before next inference
  delay(1000);
}
