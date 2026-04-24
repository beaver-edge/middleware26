#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// TensorFlow Lite globals
tflite::MicroErrorReporter tflErrorReporter;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInput = nullptr;
constexpr int tensorArenaSize = 2 * 1024;
uint8_t tensorArena[tensorArenaSize];

// Built-in RGB color sensor
APDS9960 rgbSensor(Wire, -1);  // Initialize with Wire and no interrupt pin

// Classification classes
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the color sensor
  if (!rgbSensor.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }
  
  // Load the model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version error!");
    while (1);
  }

  // Set up the resolver
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  tflInterpreter = new tflite::MicroInterpreter(tflModel, resolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory
  TfLiteStatus status = tflInterpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("Tensor allocation error!");
    while (1);
  }

  // Define model inputs
  tflInput = tflInterpreter->input(0);
  if ((tflInput->dims->size != 2) || (tflInput->dims->data[0] != 1) || (tflInput->dims->data[1] != 3)) {
    Serial.println("Invalid input dimensions!");
    while (1);
  }
}

void loop() {
  // Preprocessing: Read color sensor data
  int red, green, blue;
  if (!rgbSensor.colorAvailable() || !rgbSensor.readColor(red, green, blue)) {
    Serial.println("Error reading color sensor.");
    return;
  }

  // Normalizing sensor data
  tflInput->data.f[0] = static_cast<float>(red) / 255.0;
  tflInput->data.f[1] = static_cast<float>(green) / 255.0;
  tflInput->data.f[2] = static_cast<float>(blue) / 255.0;

  // Invoke the interpreter
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  // Process the output of model inference
  TfLiteTensor* output = tflInterpreter->output(0);
  int maxIndex = 0;
  float maxScore = 0;

  for (int i = 0; i < 3; ++i) {
    float score = output->data.f[i];
    if (score > maxScore) {
      maxScore = score;
      maxIndex = i;
    }
  }

  // Output the classified object as an emoji
  switch (maxIndex) {
    case 0:
      Serial.println("🍎"); // Apple
      break;
    case 1:
      Serial.println("🍌"); // Banana
      break;
    case 2:
      Serial.println("🍊"); // Orange
      break;
    default:
      Serial.println("?"); // Unknown
      break;
  }

  delay(1000);
}
