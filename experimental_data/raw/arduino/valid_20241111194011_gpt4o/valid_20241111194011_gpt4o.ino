#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include "model.h"  // The model is stored here

// Declare variables for TensorFlow Lite
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
uint8_t tensor_arena[2 * 1024];

// Number of output classes
const int kNumClasses = 3;
const char* kClasses[kNumClasses] = {"\U0001F34E Apple", "\U0001F34C Banana", "\U0001F34A Orange"};

void setup() {
  Serial.begin(9600);
  while (!Serial) {}

  // Setup the color sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS-9960 sensor!");
    while(1);
  }
  
  // Load the model
  tflite_model = tflite::GetModel(model);
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);

  // Allocate memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("Didn't allocate memory!");
    while(1);
  }

  // Define model input
  input = interpreter->input(0);
  if (input->dims->size != 2 || input->dims->data[0] != 1 || input->dims->data[1] != 3) {
    error_reporter->Report("Bad input tensor parameters in model");
    while(1);
  }

  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Bad input tensor type");
    while(1);
  }

  Serial.println("Setup complete. Ready to classify.");
}

void loop() {
  int red, green, blue, ambient;
  
  // Read the color sensor
  if (!APDS.colorAvailable()) {
    delay(5); // Wait for color to be available
    return;
  }

  APDS.readColor(red, green, blue, ambient);

  // Normalize the color data to [0, 1]
  input->data.f[0] = red / 65535.0;
  input->data.f[1] = green / 65535.0;
  input->data.f[2] = blue / 65535.0;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed!");
    return;
  }

  // Get the output from the inference
  TfLiteTensor* output = interpreter->output(0);
  
  int max_index = 0;
  float max_value = output->data.f[0];
  for (int i = 1; i < kNumClasses; i++) {
      if (output->data.f[i] > max_value) {
          max_value = output->data.f[i];
          max_index = i;
      }
  }

  // Print the classified object class via Serial
  Serial.print("Detected Object: ");
  Serial.println(kClasses[max_index]);

  delay(1000); // Delay between inferences
}
