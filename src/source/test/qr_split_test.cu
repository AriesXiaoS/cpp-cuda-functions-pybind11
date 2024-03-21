#include "common.h"
#include "utils.h"

#include "cuda.cuh"
#include "check.cuh"







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
                                            int max_iters, float tolerance
){
    auto buf = A.request();
    if(buf.ndim != 2){
        throw std::runtime_error("Number of dimensions must be 2");
    }
    if(buf.shape[0] != 3 || buf.shape[1] != 3){
        throw std::runtime_error("Matrix must be 3x3");
    }

    float* ptr = (float*) buf.ptr;
    auto eig_val_res = py::array_t<float>(3);
    auto eig_val_res_buf = eig_val_res.request();
    float* eig_val_res_ptr = (float*) eig_val_res_buf.ptr;

    auto eig_vec_res = py::array_t<float>(9);
    auto eig_vec_res_buf = eig_vec_res.request();
    float* eig_vec_res_ptr = (float*) eig_vec_res_buf.ptr;

    if(device < 0){
        QREigens_3x3(ptr, eig_val_res_ptr, eig_vec_res_ptr, max_iters, tolerance);
    }else{
        float* A_device, *eig_val_device, *eig_vec_device;
        cudaSetDevice(device);
        CUDA_CHECK(cudaGetLastError());
        // Allocate memory on the device
        CUDA_CHECK(cudaMalloc((void**)&A_device, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(A_device, ptr, 9 * sizeof(float), cudaMemcpyHostToDevice));
        // Allocate memory else
        CUDA_CHECK(cudaMalloc((void**)&eig_val_device, 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&eig_vec_device, 9 * sizeof(float)));
        // Call the kernel
        dim3 dimBlock(1, 1, 1);
        dim3 dimGrid(1, 1, 1);
        QREigensTestKernel_3x3<<<dimGrid, dimBlock>>>(A_device, eig_val_device, eig_vec_device, max_iters, tolerance);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // Copy the result back
        CUDA_CHECK(cudaMemcpy(eig_val_res_ptr, eig_val_device, 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(eig_vec_res_ptr, eig_vec_device, 9 * sizeof(float), cudaMemcpyDeviceToHost));
        // Free the memory
        CUDA_CHECK(cudaFree(A_device));
        CUDA_CHECK(cudaFree(eig_val_device));
    }

    eig_vec_res.resize({3,3});
    
    std::vector<py::array_t<float>> result;
    result.push_back(eig_val_res);
    result.push_back(eig_vec_res);

    return result;
}













