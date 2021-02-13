
#ifndef TENSORRTMODEL_LOGGER_H
#define TENSORRTMODEL_LOGGER_H

#include <NvInfer.h>

/**
 * Create a logger object to print errors related to TensorRT during its usage.
 */
class Logger: public nvinfer1::ILogger {

public:

    void log(Severity severity, const char* msg) override;
};


#endif
