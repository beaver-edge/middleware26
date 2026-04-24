#include <Wire.h>
#include <Arduino_APDS9960.h> // Assuming this is the library being used based on the previous code and error context.
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model.h" // The model header file generated should reflect correct buffer usage

// Constants and variables
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* flatbuffer_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

const char* kClasses[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Wire.begin();
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  // Load the TensorFlow model from the buffer
  flatbuffer_model = tflite::GetModel(model);
  if (flatbuffer_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version does not match Schema version.");
    while (1);
  }

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
    flatbuffer_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Obtain the input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int r, g, b;
  if (!APDS.colorAvailable()) {
    Serial.println("RGB data not available!");
    return;
  }

  APDS.readColor(r, g, b);

  // Preprocessing - normalize the color data
  input->data.f[0] = static_cast<float>(r) / 255.0;
  input->data.f[1] = static_cast<float>(g) / 255.0;
  input->data.f[2] = static_cast<float>(b) / 255.0;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Post-processing - get the index of the highest output value
  int class_index = -1;
  float max_value = -1;
  for (int i = 0; i < 3; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      class_index = i;
    }
  }

  // Output the results
  if (class_index != -1) {
    Serial.print("Object detected: ");
    Serial.println(kClasses[class_index]);
  } else {
    Serial.println("No object detected.");
  }

  delay(1000); // Delay for 1 second
}
