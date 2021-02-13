
#include "logger.h"

#include <iostream>
#include <NvInfer.h>


void Logger::log(Severity severity, const char* msg) {

    // Print logs in case of error only
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
        std::cout << msg << "\n";
    }
}
