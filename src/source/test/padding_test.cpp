#include "common.h"
#include "utils.h"


py::array_t<float> Padding3dTest(py::array_t<float> vec, float pad_value, int pad_size_0, int pad_size_1, int pad_size_2){
    auto buf = vec.request();
    // check dim
    if( buf.ndim != 3 ){
        std::stringstream strstr;
        strstr << "ndim of vec should be 3, but got " << buf.ndim ;
        throw std::runtime_error(strstr.str());
    }
    //
    int new_size = (buf.shape[0]+2*pad_size_0)*(buf.shape[1]+2*pad_size_1)*(buf.shape[2]+2*pad_size_2);
    auto result = py::array_t<float>(new_size);
    py::buffer_info buf_res = result.request();
    float* ptr_res = (float*)buf_res.ptr;

    float* ptr = (float*)buf.ptr;

    PaddingFlattenedArr_3D(ptr, ptr_res,
    buf.shape[0], buf.shape[1], buf.shape[2], 
    pad_value, pad_size_0, pad_size_1, pad_size_2);

    result.resize({buf.shape[0]+pad_size_0*2, buf.shape[1]+pad_size_1*2, buf.shape[2]+pad_size_2*2});
    
    return result;
}


