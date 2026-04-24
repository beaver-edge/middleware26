#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

#define TENSOR_ARENA_SIZE 2000

namespace {
  tflite::MicroErrorReporter error_reporter;
  const tflite::Model* tflite_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  uint8_t tensor_arena[TENSOR_ARENA_SIZE];
  APDS9960 apds(Wire, /*intPin=*/2);
}

void setup() {
  Serial.begin(9600);
  while(!Serial);

  // Sensor setup
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &error_reporter);

  // Allocate memory
  interpreter->AllocateTensors();

  // Define model inputs/outputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Collecting data from sensor
  int red, green, blue;
  if (!apds.readColor(red, green, blue)) {
    Serial.println("Error reading APDS9960 data.");
    return;
  }

  // Preprocess: Normalize the RGB values (0-1 range assumed)
  input->data.f[0] = (red / 65536.0);
  input->data.f[1] = (green / 65536.0);
  input->data.f[2] = (blue / 65536.0);

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error during inference.");
    return;
  }

  // Postprocess: Get the highest confidence class
  uint8_t best_index = 0;
  for (int i = 1; i < 3; i++) {
    if (output->data.uint8[i] > output->data.uint8[best_index]) {
      best_index = i;
    }
  }

  // Output class with emoji
  const char* class_name[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};
  Serial.print("Detected: ");
  Serial.println(class_name[best_index]);

  // Delay before next reading
  delay(1000);
}