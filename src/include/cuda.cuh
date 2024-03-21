#ifndef _CUDA_H
#define _CUDA_H

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h> 

#include "define.h"

/**
* @brief 三维卷积
* @param input 输入图像 padding之后的图像
* @param output 输出图像
* @param kernel 卷积核 k*k*k 立方体
* @param param  相关参数
    struct Conv3DParam{
        int kernel_size;
        int img_shape[3];
    }
*/
__global__ void CudaConv3D(float* input, float* output, 
                           float* kernel, Conv3DParam param);

/**
*  @brief Householder变换进行QR分解 A = QR
*  https://blog.csdn.net/ZHT2016iot/article/details/115448138
*  @param A  3x3  float[9]
*  @param Q  3x3  float[9]
*  @param R  3x3  float[9]
*/
__device__ __host__ void QRSplit_3x3(float* A, float* Q, float* R);

/**
*  @brief 计算3x3矩阵特征值和特征向量
*  @param A 输入矩阵 3x3  float[9]
*  @param max_iter 最大迭代次数
*  @param tolerance 迭代的下三角全0最大容忍值
*  @return eig_val float[3]  eig_vec float[9]
*/
__device__ __host__ void QREigens_3x3(float* A, 
                                float* eig_val, float* eig_vec,
                                int max_iter = 30, float tolerance = 1e-5);

// just test
__global__ void QRSplitTestKernel_3x3(float* A, float* Q, float* R);
__global__ void QREigensTestKernel_3x3(float* A, float* eig_val, float* eig_vec, 
                                        int iters = 30, float tolerance = 1e-5);


#endif 


