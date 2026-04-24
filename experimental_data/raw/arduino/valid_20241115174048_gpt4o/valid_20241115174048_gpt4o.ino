#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>
#include "./model.h"

// Initialization
#define LED_PIN 13

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 1024;
uint8_t tensor_arena[kTensorArenaSize];
APDS9960 apds(Wire, -1); // Provide TwoWire reference and an interrupt pin (e.g., -1 if not used)
const char* class_labels[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  // Initialize the LED pin as an output
  pinMode(LED_PIN, OUTPUT);

  // Initialize Serial communication
  Serial.begin(9600);

  // Initialize the sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor");
    while (1) delay(10);
  }

  // Create an error reporter
  error_reporter = nullptr; // ErrorReporter isn't used directly in the .
  
  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version differs from supported version.");
    return;
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Define model inputs
  input = interpreter->input(0);

  // Verify input dimensions
  if ((input->dims->size != 2) || (input->dims->data[0] != 1) || (input->dims->data[1] != 3)) {
    Serial.println("Input tensor has incorrect dimensions");
    return;
  }
  if (input->type != kTfLiteFloat32) {
    Serial.println("Input tensor data type is not float");
    return;
  }
}

void loop() {
  // Preprocessing: Read sensor data
  int r, g, b, c;
  if (!apds.readColor(r, g, b, c)) {
    Serial.println("Error reading color sensor values");
    return;
  }

  // Normalization if needed
  float red = r / 65535.0;
  float green = g / 65535.0;
  float blue = b / 65535.0;

  // Copy data to the model input buffer
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // Inference: Invoke the interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Postprocessing: Process and print the output
  output = interpreter->output(0);
  uint8_t class_index = output->data.uint8[0];

  // Output the class to serial
  Serial.print("Classified as: ");
  Serial.println(class_labels[class_index]);

  // Optional: Visual indication
  digitalWrite(LED_PIN, HIGH);
  delay(100);
  digitalWrite(LED_PIN, LOW);
  delay(900);
}
