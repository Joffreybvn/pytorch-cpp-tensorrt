
# PyTorch to C++ TensorRT

Transformation process of a Python Pytorch GPU model into an optimized TensorRT C++ one.

## Dependencies

- Ubuntu `18.04`
- CUDA `10.2`
- libCuDNN `7.6.5` (dev + runtimes)
- TensorRT `7.0.0`

### C++ dependencies:

- **boost**, with `sudo apt install libboost-all-dev` - [docs](https://www.boost.org/)
- **hdf5**, with `sudo apt-get install libhdf5-dev` - [C++ API](https://portal.hdfgroup.org/pages/viewpage.action?pageId=50073884), [Example: Read h5 file](https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html)
- **HighFive** `2.2.2`, with Cmake - [Github](https://github.com/BlueBrain/HighFive), [C++ API](https://bluebrain.github.io/HighFive/)
- **xtl** `0.7.2`, with CMake - [C++ API](https://xtl.readthedocs.io/en/latest/index.html)
- **xtensor** `0.23.0`, with Cmake - [Github](https://github.com/xtensor-stack/xtensor-io), [C++ API](https://xtensor.readthedocs.io/en/latest/index.html)
- **xtensor-io** dependencies, with `sudo apt-get install libsndfile-dev libopenimageio-dev zlib1g-dev`
- **xtensor-io** `0.12.1`, with Cmake - [Github](https://github.com/xtensor-stack/xtensor-io), [C++ API](https://xtensor-io.readthedocs.io/en/latest/)


### Documentaion


HDF5 dataset:
 - [Install from TAR](https://github.com/HDFGroup/hdf5/blob/develop/release_docs/INSTALL)
 - [C++ API](https://portal.hdfgroup.org/pages/viewpage.action?pageId=50073884)
 - Example: [Read HD5F file](https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html)

Xtensor (NumPy for C++):
 - https://towardsdatascience.com/linear-algebra-in-c-with-xtensor-is-just-like-numpy-1a6b1ee00736

PyTorch to TensorRT via ONNX:
 - Tutorial: [PyTorch to Onnx TensorRT Python](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)
 - Tutorial: [Onnx to TensorRT Python](https://medium.com/@fanzongshaoxing/accelerate-pytorch-model-with-tensorrt-via-onnx-d5b5164b369)
 - Example: [Onnx to TensorRT C++](https://github.com/spmallick/learnopencv/blob/master/PyTorch-ONNX-TensorRT-CPP/trt_sample.cpp)

Tensor RT documentation:
- [Developper guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-developer-guide/index.html)
- [Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [C++ API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/)
