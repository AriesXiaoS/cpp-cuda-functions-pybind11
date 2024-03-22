
#include "define.h"
#include "cuda.cuh"

// 以cuda全局内存的形式设置变量
__device__ int maxIters_d;
__device__ float tolerance_d;
__device__ int eigenVectorType_d;
__device__ int imgShape_d[3];

__global__ void SetHessianParams(
    int imgShape_0, int imgShape_1, int imgShape_2,
    int maxIters, float tolerance, int eigenVectorType)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;    
    if(z==0 && y==0 && x==0){
        maxIters_d = maxIters;
        tolerance_d = tolerance;
        eigenVectorType_d = eigenVectorType;
        imgShape_d[0] = imgShape_0;
        imgShape_d[1] = imgShape_1;
        imgShape_d[2] = imgShape_2;
    }
}

__global__ void CudaHessianEigen(SDM3D *hessian, Eigen3D *eigen, float* HFnorm)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;    
    // printf("gridDim %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
    if( z < imgShape_d[0] && y < imgShape_d[1] && x < imgShape_d[2] ){
        // printf("imgshape: %d %d %d\n", imgShape_d[0], imgShape_d[1], imgShape_d[2]);
        // printf("zyx %d %d %d\n", z, y, x);
        // if(z>200) printf("zyx %d %d %d\n", z, y, x);
        int idx = z * imgShape_d[1] * imgShape_d[2] + y * imgShape_d[2] + x;
        float* A;
        cudaMalloc((void**)&A, 9 * sizeof(float));
        // printf("idx %d  hessian %f", idx, hessian->xx[idx]);
        A[0] = hessian->xx[idx]; A[1] = hessian->xy[idx]; A[2] = hessian->xz[idx];
        A[3] = hessian->xy[idx]; A[4] = hessian->yy[idx]; A[5] = hessian->yz[idx];
        A[6] = hessian->xz[idx]; A[7] = hessian->yz[idx]; A[8] = hessian->zz[idx];
        //
        // printf("A: %f %f %f\n", A[0], A[1], A[2]);
        float* eigenValues;
        cudaMalloc((void**)&eigenValues, 3 * sizeof(float));
        float* eigenVectors;
        if(eigenVectorType_d){
            cudaMalloc((void**)&eigenVectors, 9 * sizeof(float));
        }else{
            eigenVectors = nullptr;
        }
        // QR 
        QREigens_3x3(A, eigenValues, eigenVectors, maxIters_d, tolerance_d);
        printf("QREigens done \n");
        eigen->eigenValues[idx*3    ] = eigenValues[0];
        eigen->eigenValues[idx*3 + 1] = eigenValues[1];
        eigen->eigenValues[idx*3 + 2] = eigenValues[2];
        HFnorm[idx] = sqrt(eigenValues[0]*eigenValues[0] 
                         + eigenValues[1]*eigenValues[1] 
                         + eigenValues[2]*eigenValues[2]);
        if(isnan(HFnorm[idx])){
            HFnorm[idx] = 0;
        }
        
        if(eigenVectorType_d == 1){
            for(int i = 0; i < 9; i++){
                eigen->eigenVectors[idx*9 + i] = eigenVectors[i];
            }
        }
        // else if(eigenVectorType_d == EIGENVEC::SPHERE){
            
        // }
    }
}

















