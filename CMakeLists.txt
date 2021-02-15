cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

project(TensorRTModel)
add_executable(TensorRTModel
        main.cpp
        src/utils/logger.cpp
        src/utils/logger.h
        src/utils/trt_destroy.h
        src/trt_engine.cpp
        src/trt_engine.h
)

find_package(Threads)
find_package(CUDA)

# Include TensorRT
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(TensorRTModel
        /usr/lib/x86_64-linux-gnu/libnvinfer.so
        /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
        /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so
)

