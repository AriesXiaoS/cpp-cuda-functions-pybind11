#include "common.h"
#include "utils.h"
#include "define.h"

#include "cuda.cuh"
#include "check.cuh"


template <typename T>
map<string, py::array_t<T>> HessianEigenTest_3x3(
    py::array_t<T> A, int vecType,
    int device, int maxIters, T tolerance)
{
    auto buf = A.request();
    T* ptr = (T*) buf.ptr;
    if(buf.ndim != 2){
        throw std::runtime_error("Number of dimensions must be 2");
    }
    if(buf.shape[0] != 3 || buf.shape[1] != 3){
        throw std::runtime_error("Matrix must be 3x3");
    }
    cudaSetDevice(device);
    CUDA_CHECK(cudaGetLastError());

    SDM3D<T>* hessian = new SDM3D<T>();
    SDM3D<T>* hessian_d;    
    CUDA_CHECK(cudaMalloc((void**)&hessian->xx, sizeof(T) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->yy, sizeof(T) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->zz, sizeof(T) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->xy, sizeof(T) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->xz, sizeof(T) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->yz, sizeof(T) * 2));
    CUDA_CHECK(cudaMemcpy(hessian->xx, ptr, sizeof(T) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->yy, ptr + 4, sizeof(T) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->zz, ptr + 8, sizeof(T) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->xy, ptr + 1, sizeof(T) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->xz, ptr + 2, sizeof(T) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->yz, ptr + 5, sizeof(T) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&hessian_d, sizeof(SDM3D<T>)));
    CUDA_CHECK(cudaMemcpy(hessian_d, hessian, sizeof(SDM3D<T>), cudaMemcpyHostToDevice));

    Eigen3D<T>* eigen = new Eigen3D<T>();
    Eigen3D<T>* eigen_d;
    CUDA_CHECK(cudaMalloc((void**)&eigen->eigenValues, sizeof(T) * 3));
    CUDA_CHECK(cudaMalloc((void**)&eigen->eigenVectors, sizeof(T) * 9));
    CUDA_CHECK(cudaMalloc((void**)&eigen_d, sizeof(Eigen3D<T>)));
    CUDA_CHECK(cudaMemcpy(eigen_d, eigen, sizeof(Eigen3D<T>), cudaMemcpyHostToDevice));

    T* HFnorm_d;
    CUDA_CHECK(cudaMalloc((void**)&HFnorm_d, sizeof(T) * 3));

    int* imgShape = new int[3]{1, 2, 1};
    //
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1, 1, 1);
    // SetHessianParams<<<dimGrid, dimBlock>>>(imgShape[0],  imgShape[1], imgShape[2], 
    //                                         maxIters, tolerance, vecType);
    CudaHessianEigen<T><<<dimGrid, dimBlock>>>(hessian_d, eigen_d, HFnorm_d,
            imgShape[0], imgShape[1], imgShape[2], maxIters, tolerance, vecType);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto eigenValue_pyArray = py::array_t<T>(3);
    auto eigenValue_buf = eigenValue_pyArray.request();
    T* eigenValue_ptr = (T*) eigenValue_buf.ptr;

    auto eigenVector_pyArray = py::array_t<T>(9);
    auto eigenVector_buf = eigenVector_pyArray.request();
    T* eigenVector_ptr = (T*) eigenVector_buf.ptr;

    CUDA_CHECK(cudaMemcpy(eigen, eigen_d, sizeof(Eigen3D<T>), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(eigenValue_ptr, eigen->eigenValues, sizeof(T) * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(eigenVector_ptr, eigen->eigenVectors, sizeof(T) * 9, cudaMemcpyDeviceToHost));

    eigenVector_pyArray.resize({3, 3});
    map<string, py::array_t<T>> result;
    result["eigenValues"] = eigenValue_pyArray;
    result["eigenVectors"] = eigenVector_pyArray;

    // 释放内存
    CUDA_CHECK(cudaFree(hessian->xx));
    CUDA_CHECK(cudaFree(hessian->yy));
    CUDA_CHECK(cudaFree(hessian->zz));
    CUDA_CHECK(cudaFree(hessian->xy));
    CUDA_CHECK(cudaFree(hessian->xz));
    CUDA_CHECK(cudaFree(hessian->yz));

    CUDA_CHECK(cudaFree(eigen->eigenValues));
    CUDA_CHECK(cudaFree(eigen->eigenVectors));

    delete [] imgShape;

    return result;

}   

template map<string, py::array_t<float>> HessianEigenTest_3x3<float>(
    py::array_t<float> A, int vecType,
    int device, int maxIters, float tolerance);
template map<string, py::array_t<double>> HessianEigenTest_3x3<double>(
    py::array_t<double> A, int vecType,
    int device, int maxIters, double tolerance);



