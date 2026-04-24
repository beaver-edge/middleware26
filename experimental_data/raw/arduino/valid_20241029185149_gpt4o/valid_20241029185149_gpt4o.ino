#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"  // Include the model header

// Custom error reporter class
class ArduinoErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    vprintf(format, args);
    return 0;
  }
};

// Variable declarations
ArduinoErrorReporter error_reporter;
const tflite::Model* tflite_model = nullptr; 
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define the tensor arena size
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Initialize APDS9960 with Wire interface and an interrupt pin (if required)
APDS9960 rgbSensor(Wire, 2);  // Change '2' to the actual interrupt pin you're using

// Define classes for classification
const char* classes[] = {"Apple", "Banana", "Orange"};

void setup() {
  // Set up the serial communication
  Serial.begin(9600);

  // Set up the sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }

  // Load the model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Set up the resolver and interpreter
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's input and output tensors
  interpreter->AllocateTensors();

  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Check input dimensions
  if (input->dims->size != 2 || input->dims->data[0] != 1 || input->dims->data[1] != 3) {
    Serial.println("Model input dimensions mismatch!");
    while (1);
  }
}

void loop() {
  // Read RGB values
  int r, g, b;
  if (!rgbSensor.readColor(r, g, b)) {
    Serial.println("Failed to read color sensor!");
    return;
  }

  // Normalize and copy the data to the input tensor
  input->data.f[0] = static_cast<float>(r) / 65536.0f;
  input->data.f[1] = static_cast<float>(g) / 65536.0f;
  input->data.f[2] = static_cast<float>(b) / 65536.0f;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke interpreter!");
    return;
  }

  // Process the model output and print the result
  int maxIndex = 0;
  for (int i = 1; i < 3; i++) {
    if (output->data.f[i] > output->data.f[maxIndex]) {
      maxIndex = i;
    }
  }

  // Print the classification result
  Serial.print("Detected: ");
  Serial.println(classes[maxIndex]);

  // Small delay before the next loop iteration
  delay(500);
}
