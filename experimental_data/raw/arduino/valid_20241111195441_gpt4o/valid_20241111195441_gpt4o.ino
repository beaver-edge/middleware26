#include <Arduino_LSM9DS1.h>       // Include Arduino LSM9DS1 sensor library
#include <TensorFlowLite.h>        // Include the TensorFlow Lite base library first
#include "tensorflow/lite/micro/all_ops_resolver.h" // Use all ops resolver
#include "tensorflow/lite/micro/micro_error_reporter.h" // Error reporter
#include "tensorflow/lite/micro/micro_interpreter.h"    // Micro interpreter
#include "tensorflow/lite/schema/schema_generated.h"    // Include FlatBufferModel schema
#include "tensorflow/lite/version.h"                    // Include TFLite versioning
#include "./model.h"                // Include the model header with model data

// TensorFlow Lite model variables
constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model_pointer = nullptr; // Renaming to avoid conflict
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
tflite::AllOpsResolver resolver;

// Define object classes
const char* classes[] = {"🍎 Apple", "🍌 Banana", "🍊 Orange"};

void setup() {
  Serial.begin(9600);
  while (!Serial); // Wait until the serial is ready

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Set up the error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  model_pointer = tflite::GetModel(model);  // Use the correct variable name
  if (model_pointer->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model_pointer, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Get input & output pointers for the TensorFlow Lite model
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  // Capture RGB data from the IMU sensor (using LSM9DS1)
  float ax, ay, az, gx, gy, gz, mx, my, mz;
  if (IMU.magneticFieldAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    IMU.readMagneticField(mx, my, mz);

    // Simulate RGB sensor with magnetic field readings
    float red = mx;
    float green = my;
    float blue = mz;

    // Map sensor data to model input tensor
    input->data.f[0] = red;
    input->data.f[1] = green;
    input->data.f[2] = blue;

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }

    // Read inferred class from the model's output
    uint8_t class_index = output->data.uint8[0];
    if (class_index < 3) {
      Serial.println(classes[class_index]);
    } else {
      Serial.println("Unknown class");
    }
  }

  delay(1000); // Add some delay between consecutive reads
}
