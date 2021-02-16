
#include <iostream>
#include <memory>
#include <chrono>

#include "src/trt_engine.h"

const std::string PATH_ONNX_MODEL = "../models/full_model.onnx";
const std::string PATH_IMAGE("../datasets/fruits_360/Test/Avocado/4_100.jpg");


int main() {

    // Init TensorRT engine and parse ONNX model
    TensorRTEngine engine(PATH_ONNX_MODEL);

    // Make a prediction + Time evaluation
    auto start = std::chrono::high_resolution_clock::now();
    engine.predict(PATH_IMAGE, 1);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast< std::chrono::milliseconds>(stop - start);
    std::cout << "Inference completed in: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}