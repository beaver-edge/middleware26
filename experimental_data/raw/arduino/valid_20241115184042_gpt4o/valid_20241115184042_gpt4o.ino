#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// TensorFlow Lite Micro's components
tflite::MicroErrorReporter error_reporter;
const tflite::Model* tfl_model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create a memory arena for the TensorFlow Lite framework
constexpr int kTensorArenaSize = 1024;
byte tensor_arena[kTensorArenaSize];

// APDS-9960 Color Sensor
APDS9960 rgbSensor(Wire, 0x39);  // include Wire library and set address if needed

// Classification labels
const char* classLabels[] = {"Apple", "Banana", "Orange"};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS-9960 sensor!");
    while (1);
  }

  // Load and configure the TensorFlow Lite model
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version is not supported!");
    while (1);
  }

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (1);
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int red, green, blue;
  if (rgbSensor.colorAvailable()) {
    rgbSensor.readColor(red, green, blue);

    // Pre-processing: normalize RGB values to the range [0, 1]
    float r_normalized = red / 255.0;
    float g_normalized = green / 255.0;
    float b_normalized = blue / 255.0;

    // Copy data to the input tensor
    input->data.f[0] = r_normalized;
    input->data.f[1] = g_normalized;
    input->data.f[2] = b_normalized;

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Process the output
    uint8_t max_score = 0;
    int predicted_index = -1;
    for (int i = 0; i < 3; i++) {
      if (output->data.uint8[i] > max_score) {
        max_score = output->data.uint8[i];
        predicted_index = i;
      }
    }

    // Output the prediction with a Unicode emoji
    if (predicted_index != -1) {
      Serial.print("Classification: ");
      Serial.println(classLabels[predicted_index]);
      switch (predicted_index) {
        case 0: Serial.println("🍎 Apple"); break;
        case 1: Serial.println("🍌 Banana"); break;
        case 2: Serial.println("🍊 Orange"); break;
        default: break;
      }
    }
  }

  delay(1000);
}
