#include "Arduino_APDS9960.h"
#include "Wire.h"
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "./model.h"

// Initialization
tflite::AllOpsResolver resolver;
tflite::ErrorReporter* error_reporter;
tflite::MicroInterpreter* interpreter;
const tflite::Model* model_ptr;
tflite::MicroErrorReporter micro_error_reporter;
TfLiteTensor* input;
TfLiteTensor* output;

constexpr int kTensorArenaSize = 2048;
uint8_t tensor_arena[kTensorArenaSize];

// Sensor and classification setup
APDS9960 rgbSensor(Wire, 0); // Provide TwoWire object and interrupt pin as required
const char* classes[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

void setup() {
  // Start serial communication
  Serial.begin(9600);

  // Initialize the error reporter
  error_reporter = &micro_error_reporter;

  // Load the model
  model_ptr = tflite::GetModel(model);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model version does not match Schema.");
    return;
  }

  // Instantiate the interpreter
  interpreter = new tflite::MicroInterpreter(model_ptr, resolver, tensor_arena,
                                             kTensorArenaSize, error_reporter);

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Setup sensor
  Wire.begin();
  if (!rgbSensor.begin()) {
    Serial.println("Sensor initialization failed!");
    return;
  }
  Serial.println("Sensor initialization successful");
}

void loop() {
  // Preprocessing: sensor setup
  int r, g, b;
  if (rgbSensor.readColor(r, g, b)) {
    float red = r / 255.0;
    float green = g / 255.0;
    float blue = b / 255.0;
    
    // Data copy: copy sensor data to input tensor
    input->data.f[0] = red;
    input->data.f[1] = green;
    input->data.f[2] = blue;

    // Inference: Invoke interpreter
    if (interpreter->Invoke() != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
      return;
    }

    // Postprocessing: Process the output of the model inference
    int predicted_class = output->data.uint8[0];
    Serial.println(classes[predicted_class]);
  } else {
    Serial.println("Failed to read color data from sensor");
  }

  delay(1000); // Add some delay for readability
}
