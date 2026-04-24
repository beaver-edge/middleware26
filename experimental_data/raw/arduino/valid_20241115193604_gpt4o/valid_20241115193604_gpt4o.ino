#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <ArduinoBLE.h>

#include "model.h"

// Instantiate a TensorFlow Lite error reporter
tflite::ErrorReporter* error_reporter = nullptr;

// Model setup
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena configuration
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Sensor setup
APDS9960 rgbSensor(Wire, 2);  // Assuming pin 2 is used for the interrupt pin, modify if used differently

const int input_dim = 3;
const int output_dim = 3;
const char* classes[output_dim] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

class CustomErrorReporter : public tflite::ErrorReporter {
public:
  int Report(const char* format, va_list args) override {
    vprintf(format, args);
    printf("\n");
    return 0;
  }
};

void setup() {
  // Start serial communication
  Serial.begin(9600);
  while (!Serial);  // Wait for Serial to be ready before proceeding

  // Initialize the sensor
  if (!rgbSensor.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor!");
    while (1);
  }
  Serial.println("APDS9960 sensor initialized!");

  // Load model
  tflite_model = tflite::GetModel(model);

  // Set up error reporter and ops resolver
  static CustomErrorReporter errorReporter;
  error_reporter = &errorReporter;
  
  static tflite::AllOpsResolver resolver;
  
  // Initialize interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate tensor memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Obtain input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  int r, g, b;
  
  // Read RGB values from the sensor
  if (rgbSensor.colorAvailable()) {
    rgbSensor.readColor(r, g, b);

    // Normalizing the color values
    float red = r / 65535.0;
    float green = g / 65535.0;
    float blue = b / 65535.0;

    Serial.print("R: "); Serial.print(red, 4);
    Serial.print(" G: "); Serial.print(green, 4);
    Serial.print(" B: "); Serial.println(blue, 4);

    // Fill input tensor with normalized RGB values
    input->data.f[0] = red;
    input->data.f[1] = green;
    input->data.f[2] = blue;

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }

    // Process the output and print the detected class
    uint8_t max_index = 0;
    for (int i = 1; i < output_dim; i++) {
      if (output->data.uint8[i] > output->data.uint8[max_index]) {
        max_index = i;
      }
    }
    
    // Output classified result
    Serial.print("Detected: ");
    Serial.println(classes[max_index]);
  }

  // Delay before the next read
  delay(1000);
}
