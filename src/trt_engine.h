
#ifndef TENSORRTMODEL_TRT_ENGINE_H
#define TENSORRTMODEL_TRT_ENGINE_H

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "utils/trt_destroy.h"

class TensorRTEngine {

private:

    template <class T> using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

    // Instantiate engine and context empty pointers
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};

    // Create the buffer and dimension vectors
    std::vector<void*> buffers;
    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;

    void buildEngine(const std::string& model_path);
    void allocateMemory(int batch_size);
    void releaseMemory();

    static size_t getSizeByDim(const nvinfer1::Dims& dims);

public:

    explicit TensorRTEngine(const std::string& model_path);
    void predict();
};


#endif //TENSORRTMODEL_TRT_ENGINE_H
