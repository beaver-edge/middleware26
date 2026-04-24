#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <Wire.h>

// Declare critical variables for the application
tflite::MicroErrorReporter error_reporter;
const tflite::Model* tflite_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
constexpr int kTensorArenaSize = 2 * 1024;
byte tensor_arena[kTensorArenaSize];
APDS9960 rgbSensor(Wire, -1); // Assuming no interrupt pin is used

// Initialize the sensor
void sensorSetup() {
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }
}

// Initialization phase
void setup() {
  Serial.begin(9600);
  
  // Load the model
  tflite_model = tflite::GetModel(model);  // Correct model loading
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version does not match!");
    while (1);
  }

  // Resolve operators
  tflite::AllOpsResolver resolver; // Use AllOpsResolver

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    while (1);
  }

  // Define the model inputs
  input = interpreter->input(0);
  
  // Ensure sensor is setup
  sensorSetup();
}

// Main loop
void loop() {
  int red, green, blue;

  // Read the color data
  if (!rgbSensor.colorAvailable()) {
    delay(10);
    return;
  }
  
  rgbSensor.readColor(red, green, blue);
  
  // Preprocess: Normalize the input RGB values
  float r_norm = red / 255.0;  // Assuming 8-bit RGB
  float g_norm = green / 255.0;
  float b_norm = blue / 255.0;

  // Inference: Data copy, invoke interpreter
  input->data.f[0] = r_norm;
  input->data.f[1] = g_norm;
  input->data.f[2] = b_norm;
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error: inference failed!");
    return;
  }

  // Postprocessing: Process the output
  output = interpreter->output(0);
  uint8_t class_index = output->data.uint8[0];

  // Map class index to emoji
  const char* classes[] = {"🍎", "🍌", "🍊"};
  Serial.print("Detected: ");
  Serial.println(classes[class_index]);
  
  // Small delay
  delay(1000);
}
