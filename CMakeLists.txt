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
find_package(CUDA REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package(HighFive REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-io REQUIRED)

link_directories(${HDF5_LIBRARY_DIRS})
target_include_directories(TensorRTModel PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${HighFive_INCLUDE_DIRS}
        ${xtensor_INCLUDE_DIRS}
        ${xtensor-io_INCLUDE_DIRS}
)
target_link_libraries(TensorRTModel
        HighFive
        xtensor
        xtensor-io
        /usr/lib/x86_64-linux-gnu/libnvinfer.so
        /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
        /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so
        ${HDF5_CXX_LIBRARIES}
)