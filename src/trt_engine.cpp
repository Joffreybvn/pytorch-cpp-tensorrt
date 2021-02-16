
#include <iostream>
#include <tuple>
#include <algorithm>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#include "utils/logger.h"
#include "trt_engine.h"


Logger gLogger;


TensorRTEngine::TensorRTEngine(const std::string& model_path) {

    // Build the engine and the context
    buildEngine(model_path);
};

void TensorRTEngine::predict(const std::string& image_path, int batch_size) {

    // Allocate memory and preprocess
    allocateMemory(batch_size);
    preprocessImage(image_path);

    // Do the inference
    context->enqueue(batch_size, buffers.data(), 0, nullptr);

    // Retrieve the result and release memory
    postProcess(batch_size);
    releaseMemory();
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

    // Set max batch size to 32
    builder->setMaxBatchSize(32);


    //  Create the network - Set batch size to "explicit" -> Allow dynamic batch size.
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
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

    // Set the amount of buffers (default: 2)
    buffers.push_back(malloc(sizeof(int) * engine->getNbBindings()));

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

void TensorRTEngine::preprocessImage(const std::string& image_path) {

    // Read input image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }

    // Load the image on the GPU
    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);

    auto input_width = input_dims[0].d[2];
    auto input_height = input_dims[0].d[1];
    auto channels = input_dims[0].d[0];
    auto input_size = cv::Size(input_width, input_height);

    // Normalize
    cv::cuda::GpuMat normalized_image;
    gpu_frame.convertTo(normalized_image, CV_32FC3, 1.f / 255.f);

    // Convert to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i) {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, (float*) buffers[0] + i * input_width * input_height));
    }
    cv::cuda::split(normalized_image, chw);
}

void TensorRTEngine::postProcess(int batch_size) {

    // Copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(input_dims[1]) * batch_size);
    cudaMemcpy(cpu_output.data(), buffers[1], cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

}