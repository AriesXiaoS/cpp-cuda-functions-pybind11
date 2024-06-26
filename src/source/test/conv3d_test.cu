#include "common.h"
#include "utils.h"
#include "define.h"

#include "cuda.cuh"
#include "check.cuh"


template <typename T>
py::array_t<T> CudaConv3dTest(py::array_t<T> vec, py::array_t<T> kernel, int device)
{
    auto buf = vec.request();
    auto buf_k = kernel.request();
    // check dim
    if( buf.ndim != 3 || buf_k.ndim != 3 ){
        std::stringstream strstr;
        strstr << "ndim of vec, kernel should be 3, but got " << buf.ndim << " " << buf_k.ndim ;
        throw std::runtime_error(strstr.str());
    }
    if( buf_k.shape[0] != buf_k.shape[1] || buf_k.shape[0] != buf_k.shape[2] ){
        std::stringstream strstr;
        strstr << "shape of kernel should be same, but got " << buf_k.shape[0] << " " << buf_k.shape[1] << " " << buf_k.shape[2];
        throw std::runtime_error(strstr.str());
    }
    if( buf_k.shape[0] % 2 == 0 ){
        std::stringstream strstr;
        strstr << "shape of kernel should be odd, but got " << buf_k.shape[0];
        throw std::runtime_error(strstr.str());
    }


    //
    int padding_size = (buf_k.shape[0] - 1) / 2;
    T* ptr = (T*)buf.ptr;
    T* ptr_k = (T*)buf_k.ptr;
    T* ptr_padded = new T[(buf.shape[0]+2*padding_size)*(buf.shape[1]+2*padding_size)*(buf.shape[2]+2*padding_size)];
    // padding
    PaddingFlattenedArr_3D<T>(ptr, ptr_padded,
    buf.shape[0], buf.shape[1], buf.shape[2], 
    0, padding_size, padding_size, padding_size);

    cudaSetDevice(device);
    CUDA_CHECK(cudaGetLastError());

    // 分配显存空间
    int size_bytes_padded = (buf.shape[0]+buf_k.shape[0]-1)*(buf.shape[1]+buf_k.shape[0]-1)*(buf.shape[2]+buf_k.shape[0]-1)*sizeof(T);
    int size_bytes_kernel = buf_k.shape[0]*buf_k.shape[1]*buf_k.shape[2]*sizeof(T);
    int size_bytes_res = buf.shape[0]*buf.shape[1]*buf.shape[2]*sizeof(T);
    T* cuda_vec, *cuda_kernel, *cuda_res;
    CUDA_CHECK(cudaMalloc((void**)&cuda_vec, size_bytes_padded));
    CUDA_CHECK(cudaMalloc((void**)&cuda_kernel, size_bytes_kernel));
    CUDA_CHECK(cudaMalloc((void**)&cuda_res, size_bytes_res));

    // host -> device
    CUDA_CHECK(cudaMemcpy(cuda_vec, ptr_padded, size_bytes_padded, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_kernel, ptr_k, size_bytes_kernel, cudaMemcpyHostToDevice));

    // kernel
    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid(ceil((T)buf.shape[0] / dimBlock.x), ceil((T)buf.shape[1] / dimBlock.y), ceil((T)buf.shape[2] / dimBlock.z));

    Conv3DParam param = { int(buf_k.shape[0]), 
                          {int(buf.shape[0]), int(buf.shape[1]), int(buf.shape[2])} };
    // Cuda Con v3D 
    CudaConv3D<T><<<dimGrid, dimBlock>>>(cuda_vec, cuda_res, cuda_kernel, param);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // device -> host
    T* res = new T[buf.shape[0]*buf.shape[1]*buf.shape[2]];
    CUDA_CHECK(cudaMemcpy(res, cuda_res, size_bytes_res, cudaMemcpyDeviceToHost));

    // 释放显存空间
    CUDA_CHECK(cudaFree(cuda_vec));
    CUDA_CHECK(cudaFree(cuda_kernel));
    CUDA_CHECK(cudaFree(cuda_res));

    auto result = py::array_t<T>(buf.shape[0]*buf.shape[1]*buf.shape[2]);

    py::buffer_info buf_res = result.request();
    T* ptr_res = (T*)buf_res.ptr;
    for(int i=0; i<buf.shape[0]*buf.shape[1]*buf.shape[2]; i++){
        ptr_res[i] = res[i];
    }
    result.resize({buf.shape[0], buf.shape[1], buf.shape[2]});

    return result;
}


template py::array_t<float> CudaConv3dTest(py::array_t<float> vec, py::array_t<float> kernel, int device);
template py::array_t<double> CudaConv3dTest(py::array_t<double> vec, py::array_t<double> kernel, int device);








