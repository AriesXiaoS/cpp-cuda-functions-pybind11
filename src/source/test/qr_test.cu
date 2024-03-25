#include "common.h"
#include "utils.h"

#include "cuda.cuh"
#include "check.cuh"


////////////////// TEST 
template <typename T>
__global__ void QRSplitTestKernel_3x3(T* A, T* Q, T* R){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x==0 && y==0 && z==0){
        QRSplit_3x3<T>(A, Q, R);
    }
}

template <typename T>
__global__ void QREigensTestKernel_3x3(T* A, T* eigenValues, T* eigenVectors, 
                                        int iters, T tolerance, int vecType){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if(x==0 && y==0 && z==0){
        QREigens_3x3<T>(A, eigenValues, eigenVectors, iters, tolerance, vecType);
    }
}

template <typename T>
std::vector<py::array_t<T>> QRSplitTest_3x3(py::array_t<T> A, int device){
    auto buf = A.request();

    if(buf.ndim != 2){
        throw std::runtime_error("Number of dimensions must be 2");
    }
    if(buf.shape[0] != 3 || buf.shape[1] != 3){
        throw std::runtime_error("Matrix must be 3x3");
    }

    T* ptr = (T*) buf.ptr;
    T *Q = new T[9];
    T *R = new T[9];

    if(device < 0){
        QRSplit_3x3(ptr, Q, R);
    }else{
        T* A_device, *Q_device, *R_device;
        cudaSetDevice(device);
        CUDA_CHECK(cudaGetLastError());
        // Allocate memory on the device
        CUDA_CHECK(cudaMalloc((void**)&A_device, 9 * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(A_device, ptr, 9 * sizeof(T), cudaMemcpyHostToDevice));
        // Allocate memory else
        CUDA_CHECK(cudaMalloc((void**)&Q_device, 9 * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void**)&R_device, 9 * sizeof(T)));
        // Call the kernel
        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(1, 1, 1);
        QRSplitTestKernel_3x3<T><<<dimGrid, dimBlock>>>(A_device, Q_device, R_device);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // Copy the result back
        CUDA_CHECK(cudaMemcpy(Q, Q_device, 9 * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(R, R_device, 9 * sizeof(T), cudaMemcpyDeviceToHost));
        // Free the memory
        CUDA_CHECK(cudaFree(A_device));
        CUDA_CHECK(cudaFree(Q_device));
        CUDA_CHECK(cudaFree(R_device));
    }
    

    auto Q_res = py::array_t<T>(9);
    auto Q_res_buf = Q_res.request();
    T* Q_res_ptr = (T*) Q_res_buf.ptr;

    auto R_res = py::array_t<T>(9);
    auto R_res_buf = R_res.request();
    T* R_res_ptr = (T*) R_res_buf.ptr;

    for(int i = 0; i < 9; i++){
        Q_res_ptr[i] = Q[i];
        R_res_ptr[i] = R[i];
    }

    Q_res.resize({3,3});
    R_res.resize({3,3});

    std::vector<py::array_t<T>> result;
    result.push_back(Q_res);
    result.push_back(R_res);

    return result;  

}


template <typename T>
std::vector<py::array_t<T>> QREigensTest_3x3(py::array_t<T> A, int device,
                                            int maxIters, T tolerance, int vecType)
{
    auto buf = A.request();
    if(buf.ndim != 2){
        throw std::runtime_error("Number of dimensions must be 2");
    }
    if(buf.shape[0] != 3 || buf.shape[1] != 3){
        throw std::runtime_error("Matrix must be 3x3");
    }

    T* ptr = (T*) buf.ptr;
    auto eigenValue_pyArray = py::array_t<T>(3);
    auto eigenValue_buf = eigenValue_pyArray.request();
    T* eigenValue_ptr = (T*) eigenValue_buf.ptr;

    auto eigenVector_pyArray = py::array_t<T>(9);
    auto eigenVector_buf = eigenVector_pyArray.request();
    T* eigenVector_ptr = (T*) eigenVector_buf.ptr;

    if(device < 0){
        QREigens_3x3(ptr, eigenValue_ptr, eigenVector_ptr, maxIters, tolerance, vecType);
    }else{
        T* A_device, *eigenValue_device, *eigenVector_device;
        cudaSetDevice(device);
        CUDA_CHECK(cudaGetLastError());
        // Allocate memory on the device
        CUDA_CHECK(cudaMalloc((void**)&A_device, 9 * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(A_device, ptr, 9 * sizeof(T), cudaMemcpyHostToDevice));
        // Allocate memory else
        CUDA_CHECK(cudaMalloc((void**)&eigenValue_device, 3 * sizeof(T)));
        CUDA_CHECK(cudaMalloc((void**)&eigenVector_device, 9 * sizeof(T)));
        // Call the kernel
        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(1, 1, 1);
        QREigensTestKernel_3x3<T><<<dimGrid, dimBlock>>>(A_device, eigenValue_device, eigenVector_device, maxIters, tolerance, vecType);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // Copy the result back
        CUDA_CHECK(cudaMemcpy(eigenValue_ptr, eigenValue_device, 3 * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(eigenVector_ptr, eigenVector_device, 9 * sizeof(T), cudaMemcpyDeviceToHost));
        // Free the memory
        CUDA_CHECK(cudaFree(A_device));
        CUDA_CHECK(cudaFree(eigenValue_device));
    }

    eigenVector_pyArray.resize({3,3});
    
    std::vector<py::array_t<T>> result;
    result.push_back(eigenValue_pyArray);
    result.push_back(eigenVector_pyArray);

    return result;
}

template std::vector<py::array_t<float>> QRSplitTest_3x3(py::array_t<float> A, int device);
template std::vector<py::array_t<double>> QRSplitTest_3x3(py::array_t<double> A, int device);

template std::vector<py::array_t<float>> QREigensTest_3x3(py::array_t<float> A, int device, int maxIters, float tolerance, int vecType);
template std::vector<py::array_t<double>> QREigensTest_3x3(py::array_t<double> A, int device, int maxIters, double tolerance, int vecType);













