#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model.h"  // Include the model header

// Declare necessary variables for TensorFlow Lite Micro
tflite::MicroErrorReporter error_reporter;
const tflite::Model* loaded_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Tensor arena memory allocation
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Object classes
const char* classes[] = {"🍏", "🍌", "🍊"}; // Apple, Banana, Orange

void setup() {
  Serial.begin(9600);
  
  // Initialize APDS9960 sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }
  
  // Load model
  loaded_model = tflite::GetModel(model);
  if (loaded_model == nullptr) {
    Serial.println("Model pointer is null!");
    while (1);
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(
      loaded_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

  // Allocate tensor memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read data from APDS9960 sensor
  int red, green, blue;
  if (!APDS.colorAvailable() || !APDS.readColor(red, green, blue)) {
    Serial.println("Error reading color data!");
    return;
  }

  // Normalize color values
  float norm_red = red / 255.0;
  float norm_green = green / 255.0;
  float norm_blue = blue / 255.0;

  // Copy normalized data to input tensor
  input->data.f[0] = norm_red;
  input->data.f[1] = norm_green;
  input->data.f[2] = norm_blue;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Find the class with maximum probability
  uint8_t max_index = 0;
  float max_value = output->data.f[0];
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.f[i] > max_value) {
      max_index = i;
      max_value = output->data.f[i];
    }
  }

  // Output class result
  Serial.println(classes[max_index]);

  delay(1000); // Wait for 1 second before the next reading
}
