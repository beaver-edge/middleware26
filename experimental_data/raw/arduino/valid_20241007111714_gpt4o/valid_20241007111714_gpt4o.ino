#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Ensure the model data header is available
// #include "model_data.h" // Uncomment and provide the correct path if available

// Initialization: Declare variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Define Tensor Arena
constexpr int kTensorArenaSize = 4096;
uint8_t tensor_arena[kTensorArenaSize];

// Define sensor
APDS9960 apds(Wire, 2); // Assuming pin 2 is used for the interrupt pin

void setup() {
  // Begin serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model
  // Ensure model data is included correctly and uncomment the line below
  // model = tflite::GetModel(g_model_data);
  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   error_reporter->Report("Model schema version does not match");
  //   return;
  // }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Define model inputs
  input = interpreter->input(0);

  // Sensor setup
  if (!apds.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }
  if (!apds.colorAvailable()) {
    Serial.println("Error enabling APDS9960 color sensor.");
    while (1);
  }
}

void loop() {
  // Preprocessing: Read sensor data
  int r, g, b, c;
  if (!apds.readColor(r, g, b, c)) {
    Serial.println("Error reading APDS9960 sensor data.");
    return;
  }

  // Normalize the data (based on dataset summary)
  float norm_r = r / 255.0;
  float norm_g = g / 255.0;
  float norm_b = b / 255.0;

  // Inference: Copy data to input tensor
  input->data.f[0] = norm_r;
  input->data.f[1] = norm_g;
  input->data.f[2] = norm_b;

  // Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Postprocessing: Process output
  output = interpreter->output(0);
  int class_id = output->data.uint8[0];

  // Map class ID to emoji
  const char* class_emojis[] = {"🍎", "🍌", "🍊"};
  Serial.print("Detected object: ");
  Serial.println(class_emojis[class_id]);

  delay(1000); // Delay for stability
}
