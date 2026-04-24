#include <Arduino_LSM9DS1.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <SPI.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model.h"  // The header containing the trained model

// Globals, used for TensorFlow Lite
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Pointer to the model
  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  // Input and output tensor pointers
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Memory arena for input, output, and intermediate arrays
  constexpr int kTensorArenaSize = 2048;
  uint8_t tensor_arena[kTensorArenaSize];

  // Data to store input from the color sensor
  float input_data[3];
  const char* color_classes[] = {"Apple", "Banana", "Orange"};
}

APDS9960 rgbSensor(Wire, -1); // Assume no interrupt pin is used

void setup() {
  Serial.begin(9600);
  Wire.begin();

  if (!rgbSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }

  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<2> micro_op_resolver; 
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
      tfl_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed.");
    while (1);
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
}

void loop() {
  int red, green, blue;
  if (!rgbSensor.readColor(red, green, blue)) {
    Serial.println("Failed to read color sensor!");
    return;
  }

  input_data[0] = static_cast<float>(red) / 255.0f;
  input_data[1] = static_cast<float>(green) / 255.0f;
  input_data[2] = static_cast<float>(blue) / 255.0f;

  memcpy(model_input->data.f, input_data, sizeof(input_data));

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  float* output = model_output->data.f;
  int predicted_index = -1;
  float max_value = -1.0;
  for (int i = 0; i < 3; ++i) {
    if (output[i] > max_value) {
      max_value = output[i];
      predicted_index = i;
    }
  }

  if (predicted_index >= 0 && predicted_index < 3) {
    Serial.println(color_classes[predicted_index]);
  } else {
    Serial.println("Unknown color class!");
  }
  
  delay(1000);
}
