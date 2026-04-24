#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <Arduino_APDS9960.h>
#include <Wire.h>

// Constants and model settings
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Ensure this header file with the model data array definition is present
// Uncomment the line below and provide the correct model data header file
// #include "model_data.h"  

// Ensure the model data is correctly linked (uncomment and modify if needed)
// extern const unsigned char model_data[];
const tflite::Model* model;

// Global variables for TensorFlow Lite model components
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;

// APDS9960 color sensor
APDS9960 apds(Wire, /*intPin=*/2);  // Specify the appropriate interrupt pin

// Classification labels
const char* kClasses[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

// Initialization step
void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Set up the color sensor
  Wire.begin();
  if (!apds.begin()) {
    Serial.println("Error connecting to APDS9960 sensor.");
    while (1);
  }
  
  // Load the model
  // model = tflite::GetModel(model_data);  // Uncomment and ensure model_data is defined and included
  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   Serial.println("Error: Model schema version is not compatible.");
  //   while (1);
  // }
  
  // Resolve operators
  static tflite::AllOpsResolver resolver;
  
  // Allocate interpreter
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter->AllocateTensors();
  
  // Define model input and output
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  Serial.println("Setup complete.");
}

// Main loop
void loop() {
  int r, g, b;
  
  // Collect color data
  if (!apds.readColor(r, g, b)) {
    Serial.println("Failed to read from sensor!");
    return;
  }

  // Preprocessing: Normalize and copy data to model input
  input_tensor->data.f[0] = static_cast<float>(r) / 255.0;
  input_tensor->data.f[1] = static_cast<float>(g) / 255.0;
  input_tensor->data.f[2] = static_cast<float>(b) / 255.0;
  
  // Model inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error during inference");
    return;
  }
  
  // Postprocessing: Extract the result and print emoji to Serial
  int highest_class_idx = 0;
  float highest_confidence = output_tensor->data.f[0];
  for (int i = 1; i < output_tensor->dims->data[1]; ++i) {
    if (output_tensor->data.f[i] > highest_confidence) {
      highest_confidence = output_tensor->data.f[i];
      highest_class_idx = i;
    }
  }
  
  // Output the classified object with an emoji
  Serial.println(kClasses[highest_class_idx]);

  // Delay before next reading
  delay(1000);
}
