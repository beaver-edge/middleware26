#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Model includes header
#include "model.h"

// Constants for model
const int tensorArenaSize = 2 * 1024;
uint8_t tensorArena[tensorArenaSize];

// Variables for TensorFlow Lite
tflite::MicroErrorReporter errorReporter;
const tflite::Model* modelPtr = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* modelInput = nullptr;

// Sensor instance with explicit wire and pin
APDS9960 apds(Wire, 0x39);

// Class labels
const char* classLabels[] = {"Apple", "Banana", "Orange"};

void setup() {
  // Initialize Serial for output
  Serial.begin(9600);
  while (!Serial) {} // Wait for serial to initialize

  // Initialize the APDS-9960 sensor
  if (!apds.begin()) {
    Serial.println("Failed to initialize APDS-9960 sensor!");
    while (1);
  }

  // Map the model into a usable data structure
  modelPtr = tflite::GetModel(model);
  if (modelPtr->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&errorReporter, "Model schema mismatch!");
    return;
  }

  // Set up the Op Resolver using AllOpsResolver
  static tflite::AllOpsResolver opResolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter staticInterpreter(
      modelPtr, opResolver, tensorArena, tensorArenaSize, &errorReporter);
  interpreter = &staticInterpreter;

  // Allocate memory from the tensor arena for the model's tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&errorReporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the input tensor
  modelInput = interpreter->input(0);
  if ((modelInput->dims->size != 2) ||
      (modelInput->dims->data[0] != 1) ||
      (modelInput->dims->data[1] != 3) ||
      (modelInput->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(&errorReporter, "Bad input tensor parameters in model");
    return;
  }
}

void loop() {
  int r, g, b;

  // Gather color sensor data
  if (apds.colorAvailable()) {
    apds.readColor(r, g, b);

    // Normalize and convert the sensor values to the range [0, 1] float
    modelInput->data.f[0] = r / 255.0;
    modelInput->data.f[1] = g / 255.0;
    modelInput->data.f[2] = b / 255.0;

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(&errorReporter, "Invoke failed on input");
      return;
    }

    // Determine the output and find the highest confidence index
    TfLiteTensor* output = interpreter->output(0);
    uint8_t maxIndex = 0;
    float maxConfidence = output->data.f[0];
    for (int i = 1; i < 3; i++) {
      if (output->data.f[i] > maxConfidence) {
        maxConfidence = output->data.f[i];
        maxIndex = i;
      }
    }

    // Display the classification result
    Serial.print("Detected object: ");
    Serial.println(classLabels[maxIndex]);
    delay(1000); // Wait for a second for the next reading
  }
}
