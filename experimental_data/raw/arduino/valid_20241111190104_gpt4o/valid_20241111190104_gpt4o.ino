#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h" // Include the model

// TensorFlow Lite Globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// Sensor object
APDS9960 apds(Wire, -1);  // Assuming no interrupt pin is used

// Classification labels
const char* labels[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  } else {
    Serial.println("APDS9960 initialized successfully!");
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Instantiate the Interpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate memory
  interpreter->AllocateTensors();

  // Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int r, g, b; // Use int instead of uint8_t to match function signature
  if (apds.readColor(r, g, b)) {
    // Preprocessing: normalize sensor values to [0, 1] range
    float norm_r = r / 255.0;
    float norm_g = g / 255.0;
    float norm_b = b / 255.0;

    // Data copy to model input
    input->data.f[0] = norm_r;
    input->data.f[1] = norm_g;
    input->data.f[2] = norm_b;

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Error during inference!");
      return;
    }

    // Postprocessing - get highest probability
    uint8_t highest_index = 0;
    for (int i = 1; i < 3; i++) {
      if (output->data.uint8[i] > output->data.uint8[highest_index]) {
        highest_index = i;
      }
    }

    // Output the result with Unicode emoji
    Serial.println(labels[highest_index]);
    delay(1000);
  }
}
