#include "common.h"
#include "utils.h"

#include "cuda.cuh"
#include "check.cuh"


__global__ void QRSplitTestKernel_3x3(float* A, float* Q, float* R);
__global__ void QREigensTestKernel_3x3(float* A, float* eigenValues, float* eigenVectors, 
                                        int iters = 30, float tolerance = 1e-5);


std::vector<py::array_t<float>> QRSplitTest_3x3(py::array_t<float> A, int device){
    auto buf = A.request();

    if(buf.ndim != 2){
        throw std::runtime_error("Number of dimensions must be 2");
    }
    if(buf.shape[0] != 3 || buf.shape[1] != 3){
        throw std::runtime_error("Matrix must be 3x3");
    }

    float* ptr = (float*) buf.ptr;
    float *Q = new float[9];
    float *R = new float[9];

    if(device < 0){
        QRSplit_3x3(ptr, Q, R);
    }else{
        float* A_device, *Q_device, *R_device;
        cudaSetDevice(device);
        CUDA_CHECK(cudaGetLastError());
        // Allocate memory on the device
        CUDA_CHECK(cudaMalloc((void**)&A_device, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(A_device, ptr, 9 * sizeof(float), cudaMemcpyHostToDevice));
        // Allocate memory else
        CUDA_CHECK(cudaMalloc((void**)&Q_device, 9 * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&R_device, 9 * sizeof(float)));
        // Call the kernel
        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(1, 1, 1);
        QRSplitTestKernel_3x3<<<dimGrid, dimBlock>>>(A_device, Q_device, R_device);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // Copy the result back
        CUDA_CHECK(cudaMemcpy(Q, Q_device, 9 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(R, R_device, 9 * sizeof(float), cudaMemcpyDeviceToHost));
        // Free the memory
        CUDA_CHECK(cudaFree(A_device));
        CUDA_CHECK(cudaFree(Q_device));
        CUDA_CHECK(cudaFree(R_device));
    }
    

    auto Q_res = py::array_t<float>(9);
    auto Q_res_buf = Q_res.request();
    float* Q_res_ptr = (float*) Q_res_buf.ptr;

    auto R_res = py::array_t<float>(9);
    auto R_res_buf = R_res.request();
    float* R_res_ptr = (float*) R_res_buf.ptr;

    for(int i = 0; i < 9; i++){
        Q_res_ptr[i] = Q[i];
        R_res_ptr[i] = R[i];
    }

    Q_res.resize({3,3});
    R_res.resize({3,3});

    std::vector<py::array_t<float>> result;
    result.push_back(Q_res);
    result.push_back(R_res);

    return result;  

}


std::vector<py::array_t<float>> QREigensTest_3x3(py::array_t<float> A, int device,
                                            int maxIters, float tolerance
){
    auto buf = A.request();
    if(buf.ndim != 2){
        throw std::runtime_error("Number of dimensions must be 2");
    }
    if(buf.shape[0] != 3 || buf.shape[1] != 3){
        throw std::runtime_error("Matrix must be 3x3");
    }

    float* ptr = (float*) buf.ptr;
    auto eigenValue_pyArray = py::array_t<float>(3);
    auto eigenValue_buf = eigenValue_pyArray.request();
    float* eigenValue_ptr = (float*) eigenValue_buf.ptr;

    auto eigenVector_pyArray = py::array_t<float>(9);
    auto eigenVector_buf = eigenVector_pyArray.request();
    float* eigenVector_ptr = (float*) eigenVector_buf.ptr;

    if(device < 0){
        QREigens_3x3(ptr, eigenValue_ptr, eigenVector_ptr, maxIters, tolerance);
    }else{
        float* A_device, *eigenValue_device, *eigenVector_device;
        cudaSetDevice(device);
        CUDA_CHECK(cudaGetLastError());
        // Allocate memory on the device
        CUDA_CHECK(cudaMalloc((void**)&A_device, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(A_device, ptr, 9 * sizeof(float), cudaMemcpyHostToDevice));
        // Allocate memory else
        CUDA_CHECK(cudaMalloc((void**)&eigenValue_device, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eigenVector_device, 9 * sizeof(float)));
        // Call the kernel
        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(1, 1, 1);
        QREigensTestKernel_3x3<<<dimGrid, dimBlock>>>(A_device, eigenValue_device, eigenVector_device, maxIters, tolerance);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // Copy the result back
        CUDA_CHECK(cudaMemcpy(eigenValue_ptr, eigenValue_device, 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(eigenVector_ptr, eigenVector_device, 9 * sizeof(float), cudaMemcpyDeviceToHost));
        // Free the memory
        CUDA_CHECK(cudaFree(A_device));
        CUDA_CHECK(cudaFree(eigenValue_device));
    }

    eigenVector_pyArray.resize({3,3});
    
    std::vector<py::array_t<float>> result;
    result.push_back(eigenValue_pyArray);
    result.push_back(eigenVector_pyArray);

    return result;
}





////////////////// TEST 

__global__ void QRSplitTestKernel_3x3(float* A, float* Q, float* R){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x==0 && y==0 && z==0){
        QRSplit_3x3(A, Q, R);
    }
}

__global__ void QREigensTestKernel_3x3(float* A, float* eigenValues, float* eigenVectors, 
                                        int iters, float tolerance){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x==0 && y==0 && z==0){
        QREigens_3x3(A, eigenValues, eigenVectors, iters, tolerance);
    }
}











