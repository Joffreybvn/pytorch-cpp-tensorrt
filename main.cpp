
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "src/utils/logger.h"
#include "src/utils/trt_destroy.h"


Logger logger;

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine, TRTUniquePtr<nvinfer1::IExecutionContext>& context) {

    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};

    const auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flag)};

    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

    // Parse ONNX model
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }

    // Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);

    // Use FP16 mode if possible
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // Set batch size to 1
    builder->setMaxBatchSize(1);

    // Generate TensorRT engine
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}

int main() {

    std::string PATH_ONNX_MODEL = "../models/full_model.onnx";

    // Init TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    parseOnnxModel(PATH_ONNX_MODEL, engine, context);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}