#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Declare variables for TensorFlow Lite Micro
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena size based on model requirements
constexpr int tensor_arena_size = 2048;
uint8_t tensor_arena[tensor_arena_size];

// Sensor instance (APDS9960 requires Wire and intPin parameters)
APDS9960 apds9960(Wire, 2);  // Assuming pin 2 for interrupt, change as necessary

// Classification classes
const char* classes[] = {"🍎", "🍌", "🍊"};

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the TFLite model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // Initialize the resolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Obtain model input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize APDS9960 sensor
  if (!apds9960.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while(1);
  }
}

void loop() {
  // Declare RGB values
  int red, green, blue;

  // Read RGB values from APDS9960 sensor
  if (apds9960.colorAvailable()) {
    apds9960.readColor(red, green, blue);
    
    // Normalize RGB values to [0, 1] and assign to model input
    input->data.f[0] = red / 255.0;
    input->data.f[1] = green / 255.0;
    input->data.f[2] = blue / 255.0;

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke() failed");
      return;
    }
    
    // Find output class index with the highest confidence
    uint8_t max_index = 0;
    for (uint8_t i = 1; i < 3; i++) {
      if (output->data.uint8[i] > output->data.uint8[max_index]) {
        max_index = i;
      }
    }
    
    // Print detected class as emoji
    Serial.println(classes[max_index]);
  }
  
  delay(1000); // A small delay between readings
}
