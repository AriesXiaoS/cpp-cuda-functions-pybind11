#include "common.h"

#include "cuda.cuh"
#include "check.cuh"




template <typename T>
__global__ void KernelAdd
(T *vec1, T *vec2, T* res, int num_elements){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        res[idx] = vec1[idx] + vec2[idx];
    }
}

template <typename T>
void CudaAdd(T* vec1, T* vec2, T* res, int num_elements){
    T* cuda_vec1, *cuda_vec2, *cuda_res;
    int size_bytes = num_elements*sizeof(T);
    // 分配显存空间
    CUDA_CHECK(cudaMalloc((void**)&cuda_vec1, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&cuda_vec2, size_bytes));    
    CUDA_CHECK(cudaMalloc((void**)&cuda_res, size_bytes));

    // host -> device
    CUDA_CHECK(cudaMemcpy(cuda_vec1, vec1, size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_vec2, vec2, size_bytes, cudaMemcpyHostToDevice));

    // kernel
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(ceil((T)num_elements / dimBlock.x));
    KernelAdd<T><<<dimGrid, dimBlock>>>(cuda_vec1, cuda_vec2, cuda_res, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // device -> host
    CUDA_CHECK(cudaMemcpy(res, cuda_res, size_bytes, cudaMemcpyDeviceToHost));
    
    // 释放显存空间
    CUDA_CHECK(cudaFree(cuda_vec1));
    CUDA_CHECK(cudaFree(cuda_vec2));
    CUDA_CHECK(cudaFree(cuda_res));
}


template <typename T>
py::array_t<T> AddNp(py::array_t<T> vec1, py::array_t<T> vec2, int device)
{   
    // c++ 17
    // if constexpr (std::is_same<T, int>::value) {
    //     std::cout << "Type is int" << std::endl;
    // } else if constexpr (std::is_same<T, float>::value) {
    //     std::cout << "Type is float" << std::endl;
    // } else if constexpr (std::is_same<T, double>::value) {
    //     std::cout << "Type is double" << std::endl;
    // }
    //
    auto buf1 = vec1.request(), buf2 = vec2.request();
    // check dim
    if( buf1.ndim != buf2.ndim ){
        std::stringstream strstr;
        strstr << "ndim of vec1, vec2, res should be same, but got " << buf1.ndim << " " << buf2.ndim ;
        throw std::runtime_error(strstr.str());
    }
    // check shape
    if( buf1.size != buf2.size ){
        std::stringstream strstr;
        strstr << "size of vec1, vec2, res should be same, but got " << buf1.size << " " << buf2.size ;
        throw std::runtime_error(strstr.str());
    }
    //
    auto result = py::array_t<T>(buf1.size);
    py::buffer_info buf3 = result.request();

    //获取numpy.ndarray 数据指针
    T* ptr1 = (T*)buf1.ptr;
    T* ptr2 = (T*)buf2.ptr;
    T* ptr3 = (T*)buf3.ptr;

    if(device==-1){
        // use cpu
        //申请空间
        for (int i = 0; i < buf1.size; i++)
        {
            ptr3[i] = ptr1[i] + ptr2[i];
        }
    }else{        
        cudaSetDevice(device);
        CUDA_CHECK(cudaGetLastError());
        CudaAdd<T>(ptr1, ptr2, ptr3, buf1.size);
    }
    result.resize(buf1.shape);
    return result;
}


template py::array_t<float> AddNp<float>(py::array_t<float> vec1, py::array_t<float> vec2, int device);
template py::array_t<double> AddNp<double>(py::array_t<double> vec1, py::array_t<double> vec2, int device);








