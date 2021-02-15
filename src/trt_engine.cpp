
#include <iostream>
#include <memory>
#include <tuple>
#include <algorithm>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include "utils/logger.h"
#include "trt_engine.h"


Logger gLogger;


TensorRTEngine::TensorRTEngine(const std::string& model_path) {

    // Build the engine and the context
    buildEngine(model_path);

    // Set the amount of buffers (default: 2)
    buffers[engine->getNbBindings()];
};

void TensorRTEngine::predict() {
    context->enqueue(0, buffers.data(), 0, nullptr);
};


void TensorRTEngine::buildEngine(const std::string& model_path) {

    // Create the builder and its configuration object
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

    // Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);

    // Use FP16 mode if possible
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }


    //  Create the network - Set batch size to "explicit" -> Allow dynamic batch size.
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // Create and return the network
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};


    // Create a parser on the network
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};

    // Load and parse an ONNX model
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }


    // Set the TensorRT engine and context
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
};

void TensorRTEngine::allocateMemory(int batch_size) {

    for (size_t i = 0; i < engine->getNbBindings(); ++i) {

        // Allocate the binding size
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);

        // Get input and output shape
        if (engine->bindingIsInput(i)) {
            input_dims.emplace_back(engine->getBindingDimensions(i));

        } else {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
}

void TensorRTEngine::releaseMemory() {

    // Release all buffers
    for (void* buf : buffers) {
        cudaFree(buf);
    }
}

size_t TensorRTEngine::getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;

    for (size_t i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}
