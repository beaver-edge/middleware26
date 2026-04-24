#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/kernels/micro_ops.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// TensorFlow Lite variables
tflite::MicroErrorReporter tflErrorReporter;
const tflite::Model* tflModel = nullptr;
tflite::AllOpsResolver tflOpsResolver;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 2048;
byte tensorArena[kTensorArenaSize];

// Class labels
const char* label_names[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

// Sensor
APDS9960 apds(Wire, -1);

// Initialization
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the sensor
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }
  Serial.println("APDS9960 sensor initialized.");

  // Load the model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version error.");
    while (1);
  }

  // Instantiate the Interpreter
  tflInterpreter = new tflite::MicroInterpreter(
    tflModel, tflOpsResolver, tensorArena, kTensorArenaSize, &tflErrorReporter);

  // Allocate memory for tensors
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Error allocating tensors.");
    while (1);
  }

  // Define Model Inputs
  input = tflInterpreter->input(0);
  output = tflInterpreter->output(0);
}

// Main loop
void loop() {
  int r, g, b, c;
  
  // Read the RGB values
  if (!apds.colorAvailable()) {
    delay(100);
    return;
  }
  if (!apds.readColor(r, g, b, c)) {
    Serial.println("Error reading color");
    return;
  }

  // Normalize values by dividing by 255 (assuming maximum raw value)
  input->data.f[0] = r / 255.0;
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Invoke the interpreter
  TfLiteStatus invoke_status = tflInterpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the interpreter.");
    return;
  }

  // Process Output
  uint8_t max_index = 0;
  for (uint8_t i = 1; i < 3; ++i) {
    if (output->data.uint8[i] > output->data.uint8[max_index]) {
      max_index = i;
    }
  }

  // Output the class
  Serial.print("Detected Object: ");
  Serial.println(label_names[max_index]);

  // Small delay before the next loop
  delay(500);
}
