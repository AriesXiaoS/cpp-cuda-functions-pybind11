
#include "define.h"
#include "cuda.cuh"


template <typename T>
__global__ void CudaHessianEigen(SDM3D<T> *hessian, Eigen3D<T> *eigen, T* HFnorm,
                int imgShape_0, int imgShape_1, int imgShape_2, 
                int maxIters, T tolerance, int eigenVectorType)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;    
    if( z < imgShape_0 && y < imgShape_1 && x < imgShape_2 ){
        int idx = z * imgShape_1 * imgShape_2 + y * imgShape_2 + x;
        T A[9] = {hessian->xx[idx], hessian->xy[idx], hessian->xz[idx],
                  hessian->xy[idx], hessian->yy[idx], hessian->yz[idx],
                  hessian->xz[idx], hessian->yz[idx], hessian->zz[idx]};
        T eigenValues[3];
        T eigenVectors[9];
        // QR 
        QREigens_3x3<T>(A, eigenValues, eigenVectors, maxIters, tolerance, eigenVectorType);

        eigen->eigenValues[idx*3    ] = eigenValues[0];
        eigen->eigenValues[idx*3 + 1] = eigenValues[1];
        eigen->eigenValues[idx*3 + 2] = eigenValues[2];
        HFnorm[idx] = sqrt(eigenValues[0]*eigenValues[0] 
                         + eigenValues[1]*eigenValues[1] 
                         + eigenValues[2]*eigenValues[2]);

        if(isnan(HFnorm[idx])){
            HFnorm[idx] = 0;
        }
        
        if(eigenVectorType == VEC_TYPE_CARTESIAN){
            for(int i = 0; i < 9; i++){
                eigen->eigenVectors[idx*9 + i] = eigenVectors[i];
            }
        }
    }
}




template __global__ void CudaHessianEigen<float>(SDM3D<float> *hessian, 
                Eigen3D<float> *eigen, float* HFnorm,
                int imgShape_0, int imgShape_1, int imgShape_2, 
                int maxIters, float tolerance, int eigenVectorType);
template __global__ void CudaHessianEigen<double>(SDM3D<double> *hessian, 
                Eigen3D<double> *eigen, double* HFnorm,
                int imgShape_0, int imgShape_1, int imgShape_2, 
                int maxIters, double tolerance, int eigenVectorType);













