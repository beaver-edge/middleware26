#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <Arduino_APDS9960.h>
#include "model.h"

// TensorFlow Lite setup
#define MODEL_SIZE 2 * 1024
byte tensor_arena[MODEL_SIZE];

// Error reporter
tflite::ErrorReporter* error_reporter = nullptr;

// TensorFlow Lite model pointer
const tflite::Model* tfl_model = nullptr;

// Interpreter to execute model
tflite::MicroInterpreter* interpreter = nullptr;

// Model input and output
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Operator resolver
tflite::AllOpsResolver resolver;

// RGB sensor setup
APDS9960 rgbSensor(Wire, 2);  // Initialize with appropriate wire and interrupt pin

// Classification labels
const char* classes[] = {"Apple 🍎", "Banana 🍌", "Orange 🍊"};

void setup() {
    // Initialize Serial communication
    Serial.begin(9600);
    
    // Initialize RGB Sensor
    if (!rgbSensor.begin()) {
        Serial.println("Error initializing APDS9960 sensor");
        while (1);
    }
  
    // Load the model
    tfl_model = tflite::GetModel(model);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version does not match schema");
        while (1);
    }
    
    // Set up the interpreter
    static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, MODEL_SIZE, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory for the tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors");
        while (1);
    }

    // Assign model input and output
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Setup complete");
}

void loop() {
    // Check if color data is available
    if (rgbSensor.colorAvailable()) {
        // Read RGB values
        int r, g, b;
        rgbSensor.readColor(r, g, b);

        // Preprocess: Normalize RGB values
        input->data.f[0] = r / 255.0f;
        input->data.f[1] = g / 255.0f;
        input->data.f[2] = b / 255.0f;

        // Model inference
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Invoke failed on input.");
            return;
        }

        // Postprocess: Read model output and determine the classification
        uint8_t maxIndex = 0;
        float maxValue = output->data.f[0];

        for (uint8_t i = 1; i < 3; i++) {
            if (output->data.f[i] > maxValue) {
                maxValue = output->data.f[i];
                maxIndex = i;
            }
        }

        // Print the classification result
        Serial.print("Object classified as: ");
        Serial.println(classes[maxIndex]);
    }

    // Delay before next iteration
    delay(1000);
}
