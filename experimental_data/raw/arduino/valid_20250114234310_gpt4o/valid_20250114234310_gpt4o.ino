#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h> 
#include <tensorflow/lite/micro/all_ops_resolver.h> 
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

#define NUM_CLASSES 3
const char* classLabels[NUM_CLASSES] = {"\U0001F34E", "\U0001F34C", "\U0001F34A"}; // Apple, Banana, Orange

// TensorFlow Lite micro library declarations
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
uint8_t tensor_arena[2048];
tflite::AllOpsResolver resolver;

APDS9960 rgbSensor(Wire, -1); // Initialize with Wire object and no interrupt pin

void setup() {
  Serial.begin(9600);
  
  // Wait for serial monitor to open
  while (!Serial);

  // Setup error reporting
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model from the binary data
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }
  
  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }
  
  // Define model inputs
  input = interpreter->input(0);
  if ((input->dims->data[0] != 1) || (input->dims->data[1] != 3) || (input->type != kTfLiteFloat32)) {
    Serial.println("Model input shape/type mismatch!");
    return;
  }

  // Initialize the RGB sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }
  delay(500); // a short delay to ensure everything is set up
}

void loop() {
  // Read RGB values from the sensor
  int r, g, b; // Use int to match the method signature in the library
  if (rgbSensor.colorAvailable()) {
    rgbSensor.readColor(r, g, b);

    // Preprocessing: Normalize RGB values to 0-1 range
    float r_normalized = (float)r / 255.0;
    float g_normalized = (float)g / 255.0;
    float b_normalized = (float)b / 255.0;

    // Inference: Copy normalized values to input tensor
    input->data.f[0] = r_normalized;
    input->data.f[1] = g_normalized;
    input->data.f[2] = b_normalized;

    // Invoke interpreter
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed");
      return;
    }

    // Post-processing: Get the output and determine the class
    TfLiteTensor* output = interpreter->output(0);
    uint8_t predictedClass = 0;
    int maxScore = 0; // Changed to int to match output type

    for (uint8_t i = 0; i < NUM_CLASSES; i++) {
      if (output->data.uint8[i] > maxScore) {
        maxScore = output->data.uint8[i];
        predictedClass = i;
      }
    }

    Serial.print("Predicted class: ");
    Serial.println(classLabels[predictedClass]);
  }

  delay(1000); // Delay between inferences
}
