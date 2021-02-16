
# PyTorch to C++ TensorRT

Transformation process of a Python Pytorch GPU model into an optimized C++ ONNX TensorRT one.

## Process

### Step 1: Train a model

Based on the [Fruits 360](https://www.kaggle.com/moltean/fruits) dataset, a model is trained with Pytorch GPU.

See the full process on:
 - Notebook 1: [Dataset preprocessing](./notebooks/1_dataset_preprocessing.ipynb)
 - Notebook 2: [Model creation](./notebooks/2_model_creation.ipynb)

### Step 2: Convert to ONNX

The PyTorch model is evaluated. It is then converted to an ONNX model and evaluated again to ensure that its conversion does not degrade its performance.

See the full process on:
- Notebook 3: [Pytorch model evaluation](./notebooks/3_model_evaluation.ipynb)
- Notebook 4: [ONNX convertion and evaluation](./notebooks/4_model_onnx_conversion.ipynb)

### Step 3: Deploy on TensorRT

The ONNX model is finally deployed on the GPU via TensorRT, where it is evaluated again.

See the full process on:
- Notebook 5: [TensorRT in Python](./notebooks/5_model_tensorrt.ipynb)

### Step 4: Transcript to C++

C++ transcription follows the same process as notebook 5 "[TensorRT in Python](./notebooks/5_model_tensorrt.ipynb)". 

## Documentaion

PyTorch to TensorRT via ONNX:
- Tutorial: [PyTorch to Onnx TensorRT Python](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)
- Tutorial: [Onnx to TensorRT Python](https://medium.com/@fanzongshaoxing/accelerate-pytorch-model-with-tensorrt-via-onnx-d5b5164b369)
- Example: [Onnx to TensorRT C++](https://github.com/spmallick/learnopencv/blob/master/PyTorch-ONNX-TensorRT-CPP/trt_sample.cpp)

## Dependencies

- Ubuntu `18.04`
- CUDA `10.2`
- libCuDNN `7.6.5` (dev + runtimes)
- TensorRT `7.0.0` - [Developper guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-developer-guide/index.html), [Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/), [C++ API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/)

### C++ dependencies:

- OpenCV `4.5.1` (lib + contrib) - [C++ installation guide](https://learnopencv.com/install-opencv-4-on-ubuntu-18-04/)

### Python dependencies:

The following package can be installed with `pip install`

- **torch** and **torchvision**
- **opencv-python**
- **onnx**

<details>
  <summary>Other useful libraries</summary>

These very useful libraries are not used in this project, but could be interesting for data preprocessing.

- **boost**, with `sudo apt install libboost-all-dev` - [docs](https://www.boost.org/)
- **hdf5**, with `sudo apt-get install libhdf5-dev` - [C++ API](https://portal.hdfgroup.org/pages/viewpage.action?pageId=50073884), [Read example](https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html)
- **HighFive** `2.2.2`, with Cmake - [Github](https://github.com/BlueBrain/HighFive), [C++ API](https://bluebrain.github.io/HighFive/)
- **xtl** `0.7.2`, with CMake - [C++ API](https://xtl.readthedocs.io/en/latest/index.html)
- **xtensor** `0.23.0`, with Cmake - [Github](https://github.com/xtensor-stack/xtensor-io), [C++ API](https://xtensor.readthedocs.io/en/latest/index.html), [Tutorial](https://towardsdatascience.com/linear-algebra-in-c-with-xtensor-is-just-like-numpy-1a6b1ee00736)
- **xtensor-io** dependencies, with `sudo apt-get install libsndfile-dev libopenimageio-dev zlib1g-dev`
- **xtensor-io** `0.12.1`, with Cmake - [Github](https://github.com/xtensor-stack/xtensor-io), [C++ API](https://xtensor-io.readthedocs.io/en/latest/)
</details>
