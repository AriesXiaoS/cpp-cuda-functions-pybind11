#include "common.h"
#include "utils.h"
#include "define.h"

#include "cuda.cuh"
#include "check.cuh"



map<string, py::array_t<float>> HessianEigenTest_3x3(
    py::array_t<float> A, int vecType,
    int device, int maxIters, float tolerance)
{
    auto buf = A.request();
    float* ptr = (float*) buf.ptr;
    if(buf.ndim != 2){
        throw std::runtime_error("Number of dimensions must be 2");
    }
    if(buf.shape[0] != 3 || buf.shape[1] != 3){
        throw std::runtime_error("Matrix must be 3x3");
    }
    cudaSetDevice(device);
    CUDA_CHECK(cudaGetLastError());

    SDM3D* hessian = new SDM3D();
    SDM3D* hessian_d;    
    CUDA_CHECK(cudaMalloc((void**)&hessian->xx, sizeof(float) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->yy, sizeof(float) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->zz, sizeof(float) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->xy, sizeof(float) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->xz, sizeof(float) * 2));
    CUDA_CHECK(cudaMalloc((void**)&hessian->yz, sizeof(float) * 2));
    CUDA_CHECK(cudaMemcpy(hessian->xx, ptr, sizeof(float) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->yy, ptr + 4, sizeof(float) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->zz, ptr + 8, sizeof(float) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->xy, ptr + 1, sizeof(float) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->xz, ptr + 2, sizeof(float) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hessian->yz, ptr + 5, sizeof(float) * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&hessian_d, sizeof(SDM3D)));
    CUDA_CHECK(cudaMemcpy(hessian_d, hessian, sizeof(SDM3D), cudaMemcpyHostToDevice));

    Eigen3D* eigen = new Eigen3D();
    Eigen3D* eigen_d;
    CUDA_CHECK(cudaMalloc((void**)&eigen->eigenValues, sizeof(float) * 3));
    CUDA_CHECK(cudaMalloc((void**)&eigen->eigenVectors, sizeof(float) * 9));
    CUDA_CHECK(cudaMalloc((void**)&eigen_d, sizeof(Eigen3D)));
    CUDA_CHECK(cudaMemcpy(eigen_d, eigen, sizeof(Eigen3D), cudaMemcpyHostToDevice));

    float* HFnorm_d;
    CUDA_CHECK(cudaMalloc((void**)&HFnorm_d, sizeof(float) * 3));

    int* imgShape = new int[3]{1, 2, 1};
    //
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1, 1, 1);
    SetHessianParams<<<dimGrid, dimBlock>>>(imgShape[0],  imgShape[1], imgShape[2], 
                                            maxIters, tolerance, vecType);
    CudaHessianEigen<<<dimGrid, dimBlock>>>(hessian_d, eigen_d, HFnorm_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto eigenValue_pyArray = py::array_t<float>(3);
    auto eigenValue_buf = eigenValue_pyArray.request();
    float* eigenValue_ptr = (float*) eigenValue_buf.ptr;

    auto eigenVector_pyArray = py::array_t<float>(9);
    auto eigenVector_buf = eigenVector_pyArray.request();
    float* eigenVector_ptr = (float*) eigenVector_buf.ptr;

    CUDA_CHECK(cudaMemcpy(eigen, eigen_d, sizeof(Eigen3D), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(eigenValue_ptr, eigen->eigenValues, sizeof(float) * 3, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(eigenVector_ptr, eigen->eigenVectors, sizeof(float) * 9, cudaMemcpyDeviceToHost));

    eigenVector_pyArray.resize({3, 3});
    map<string, py::array_t<float>> result;
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





