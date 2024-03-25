#include "common.h"
#include "utils.h"


template <typename T>
py::array_t<T> padding3dNp(py::array_t<T> vec, T pad_value, int pad_size_0, int pad_size_1, int pad_size_2){
    auto buf = vec.request();
    // check dim
    if( buf.ndim != 3 ){
        std::stringstream strstr;
        strstr << "ndim of vec should be 3, but got " << buf.ndim ;
        throw std::runtime_error(strstr.str());
    }
    //
    int new_size = (buf.shape[0]+2*pad_size_0)*(buf.shape[1]+2*pad_size_1)*(buf.shape[2]+2*pad_size_2);
    auto result = py::array_t<T>(new_size);
    py::buffer_info buf_res = result.request();
    T* ptr_res = (T*)buf_res.ptr;

    T* ptr = (T*)buf.ptr;

    PaddingFlattenedArr_3D<T>(ptr, ptr_res,
    buf.shape[0], buf.shape[1], buf.shape[2], 
    pad_value, pad_size_0, pad_size_1, pad_size_2);

    result.resize({buf.shape[0]+pad_size_0*2, buf.shape[1]+pad_size_1*2, buf.shape[2]+pad_size_2*2});
    
    return result;
}

template py::array_t<float> padding3dNp(py::array_t<float> vec, float pad_value, int pad_size_0, int pad_size_1, int pad_size_2);
template py::array_t<double> padding3dNp(py::array_t<double> vec, double pad_value, int pad_size_0, int pad_size_1, int pad_size_2);



