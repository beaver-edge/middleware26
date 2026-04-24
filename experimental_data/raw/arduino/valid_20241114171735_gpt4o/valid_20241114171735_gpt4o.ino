#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "model.h"

// TensorFlow Lite variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
uint8_t tensor_arena[2 * 1024];

// RGB Sensor
APDS9960 rgbSensor(Wire, /*intPin=*/2);  // Specify the interrupt pin correctly

// Classification Emojis
const char* CLASS_EMOJIS[] = {"🍎", "🍌", "🍊"};

// ErrorReporter implementation
class ArduinoErrorReporter : public tflite::ErrorReporter {
  public:
    int Report(const char* format, va_list args) override {
      vprintf(format, args);
      return 0;
    }
};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }

  // Load the TensorFlow Lite model
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version does not match Schema version!");
    while (1);
  }

  // Set up the error reporter
  static ArduinoErrorReporter arduino_error_reporter;
  error_reporter = &arduino_error_reporter;

  // In addition to an op resolver, we need a model interpreter, and memory to
  // use for input, output, and intermediate arrays.
  // The resolver is used to stitch together the ops that the models needs and
  // the interpreter uses this, as well as the model, to allocate the memory
  // needed for the model.
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Get information about the input tensor
  input_tensor = interpreter->input(0);
}

void loop() {
  int red, green, blue;

  // Read the color sensor values
  if (!rgbSensor.colorAvailable()) {
    delay(100);
    return;
  }
  rgbSensor.readColor(red, green, blue);

  // Normalize and map the sensor values to [0, 1]
  input_tensor->data.f[0] = static_cast<float>(red) / 255.0;
  input_tensor->data.f[1] = static_cast<float>(green) / 255.0;
  input_tensor->data.f[2] = static_cast<float>(blue) / 255.0;

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Get the output from the inference
  TfLiteTensor* output = interpreter->output(0);

  // Find the index of the highest confidence value
  uint8_t highest_index = 0;
  for (uint8_t i = 1; i < 3; i++) {
    if (output->data.f[i] > output->data.f[highest_index]) {
      highest_index = i;
    }
  }

  // Output the classified emoji
  Serial.println(CLASS_EMOJIS[highest_index]);

  // Delay between inferences to allow sensor to refresh
  delay(500);
}
