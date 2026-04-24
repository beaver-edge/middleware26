#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// TensorFlow Lite model variables
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
const tflite::Model* tflite_model;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// RGB sensor
APDS9960 rgbSensor(Wire, 4);  // Assuming interrupt pin 4

// Classes
const char* class_names[] = {"Apple 🍏", "Banana 🍌", "Orange 🍊"};

void setup() {
  // Initialize serial
  Serial.begin(9600);
  while (!Serial);

  // Initialize sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960.");
    while (1);
  }
  
  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }
  
  // Create the interpreter using the model, resolver, tensor arena, and error reporter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

  // Allocate memory from the tensor arena for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation error!");
    return;
  }

  // Get pointers to the input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int r, g, b;

  // Read RGB data from sensor
  if (!rgbSensor.readColor(r, g, b)) {
    Serial.println("Failed to read data from the sensor!");
    return;
  }
  
  // Normalize and fill input tensor data
  input->data.f[0] = static_cast<float>(r) / 65535.0;
  input->data.f[1] = static_cast<float>(g) / 65535.0;
  input->data.f[2] = static_cast<float>(b) / 65535.0;

  // Invoke the model
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Determine the classification result
  int classIdx = 0;
  float max_val = 0.0;
  for (int i = 0; i < 3; i++) {
    if (output->data.f[i] > max_val) {
      max_val = output->data.f[i];
      classIdx = i;
    }
  }

  // Print the result
  Serial.println(class_names[classIdx]);
  
  // Delay to allow for sensor reading and processing time
  delay(1000);
}
