
#include <iostream>
#include <memory>
#include <algorithm>

#include "src/trt_engine.h"


int main() {

    // Init TensorRT engine and parse ONNX model
    std::string PATH_ONNX_MODEL = "../models/full_model.onnx";
    TensorRTEngine engine(PATH_ONNX_MODEL);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}