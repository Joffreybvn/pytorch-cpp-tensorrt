
#include <iostream>
#include <memory>
#include <algorithm>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-io/xhighfive.hpp>

#include "H5Cpp.h"

#include "src/trt_engine.h"

const H5std_string FILE_NAME( "../datasets/dataset.h5py" );
const H5std_string DATASET_NAME( "X_test" );

int main() {

    H5::H5File file(FILE_NAME,H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(DATASET_NAME);

    if(dataset.getTypeClass() == H5T_FLOAT) {

        H5::FloatType float_type = dataset.getFloatType();

        H5std_string order_string;
        H5T_order_t order = float_type.getOrder( order_string );
        std::cout << order_string << std::endl;

        size_t size = float_type.getSize();
        std::cout << "Data size is " << size << std::endl;
    }

    H5::DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();

    hsize_t dims_out[2];
    int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
    std::cout << "rank " << rank << ", dimensions " <<
         (unsigned long)(dims_out[0]) << " x " <<
         (unsigned long)(dims_out[1]) << std::endl;


    // Init TensorRT engine and parse ONNX model
    //std::string PATH_ONNX_MODEL = "../models/full_model.onnx";
    //TensorRTEngine engine(PATH_ONNX_MODEL);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}