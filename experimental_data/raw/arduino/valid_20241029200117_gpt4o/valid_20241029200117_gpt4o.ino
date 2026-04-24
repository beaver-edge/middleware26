#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// Define constants and variables
#define TENSOR_ARENA_SIZE 4096
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// Instantiate the APDS9960
APDS9960 apds(Wire, -1); // Assuming Wire is the default I2C instance

// Object classes
const char* classes[] = {"🍎", "🍌", "🍊"};

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize APDS9960 sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor!");
    while (1);
  }
  Serial.println("APDS9960 sensor initialized.");

  // Set up the model
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.",
                           tfl_model->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Initialize the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Get pointers to the input and output tensors
  input = interpreter->input(0);
}

void loop() {
  // Read the color data
  int r, g, b;
  if (!apds.readColor(r, g, b)) {
    Serial.println("Error reading color data!");
    return;
  }

  // Normalize the color values
  float red = r / 65535.0;
  float green = g / 65535.0;
  float blue = b / 65535.0;

  // Copy the normalized color data into the input tensor
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Get the output from the model
  TfLiteTensor* output = interpreter->output(0);
  uint8_t predicted_class = output->data.uint8[0];

  // Print the classified object
  Serial.print("Predicted object: ");
  Serial.println(classes[predicted_class]);

  // Delay before next reading
  delay(1000);
}
