#ifndef _CUDA_H
#define _CUDA_H

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h> 

#include "define.h"

//

/**
* @brief 三维卷积
* @param input 输入图像 padding之后的图像
* @param output 输出图像
* @param kernel 卷积核 k*k*k 立方体
* @param param  相关参数
    struct Conv3DParam{
        int kernel_size;
        int img_shape[3]; 未padding的shape
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
*  @return eigenValues float[3]  eigenVectors float[9]
*/
__device__ __host__ void QREigens_3x3(float* A, 
                                float* eigenValues, float* eigenVectors,
                                int maxIters, float tolerance);

/**
*  @brief Hessian矩阵特征值计算
*/
__global__ void CudaHessianEigen(SDM3D *hessian, Eigen3D *eigen, float* HFnorm);
// 设置Hessian相关变量  
__global__ void SetHessianParams(
    int imgShape_0, int imgShape_1, int imgShape_2,
    int maxIters, float tolerance, int eigenVectorType);
#endif 

__global__ void CudaFrangiVo(Eigen3D *eigen, float* output);
// 设置Frangi相关变量
__global__ void SetFrangiParams(
    int imgShape_0, int imgShape_1, int imgShape_2,
    float alpha, float beta, float c, bool black_ridges);
    