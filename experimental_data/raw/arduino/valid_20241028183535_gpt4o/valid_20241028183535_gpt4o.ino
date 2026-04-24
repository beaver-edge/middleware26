#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// Global variables for TensorFlow Lite
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflite_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Create an area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 2 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// APDS-9960 sensor
APDS9960 rgbSensor(Wire, -1); // Assuming no interrupt pin is used

// Object classes
const char* classes[] = {"🍎", "🍌", "🍊"};

// Custom error reporter class
class CustomErrorReporter : public tflite::ErrorReporter {
public:
  int Report(const char* format, va_list args) override {
    vprintf(format, args);
    return 0;
  }
};

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize the sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS-9960 sensor!");
    while (1);
  }

  // Set up logging.
  static CustomErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure.
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal "
                           "to supported version %d.\n",
                           tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed\n");
    return;
  }

  // Get information about the model's input and output.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Read RGB values from the sensor
  int r, g, b;
  if (!rgbSensor.readColor(r, g, b)) {
    error_reporter->Report("Failed to read from APDS-9960 sensor!\n");
    return;
  }

  // Normalize sensor data
  input->data.f[0] = r / 255.0;
  input->data.f[1] = g / 255.0;
  input->data.f[2] = b / 255.0;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
    return;
  }

  // Get the predicted class
  float max_value = -1;
  int predicted_index = -1;
  for (int i = 0; i < 3; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      predicted_index = i;
    }
  }

  if (predicted_index != -1) {
    Serial.println(classes[predicted_index]);
  } else {
    Serial.println("Unknown class");
  }

  delay(1000);  // Delay for a while
}
