cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

project(TensorRTModel)
add_executable(TensorRTModel main.cpp src/utils/logger.cpp src/utils/logger.h src/utils/trt_destroy.h)

find_package(Threads)
find_package(CUDA REQUIRED)

# Include TensorRT
target_link_libraries(TensorRTModel /usr/lib/x86_64-linux-gnu/libnvinfer.so)
target_link_libraries(TensorRTModel /usr/lib/x86_64-linux-gnu/libnvonnxparser.so)

