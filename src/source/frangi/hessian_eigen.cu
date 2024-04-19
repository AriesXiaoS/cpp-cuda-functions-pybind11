
#include "define.h"
#include "cuda.cuh"


template <typename T>
__device__ T angleAdjust(T ang){
    // between 0,PI
    if (ang < 0){
        ang += PI;
    }else if (ang > PI){
        ang -= PI;
    }
    return ang;
}


template <typename T>
__global__ void CudaHessianEigen(SDM3D<T> *hessian, Eigen3D<T> *eigen, T* HFnorm,
                int imgShape_0, int imgShape_1, int imgShape_2, 
                int maxIters, T tolerance, int vecType)
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
        int vecSize = getVecSize(vecType);
        // QR 
        QREigens_3x3<T>(A, eigenValues, eigenVectors, maxIters, tolerance, vecSize);

        eigen->eigenValues[idx*3    ] = eigenValues[0];
        eigen->eigenValues[idx*3 + 1] = eigenValues[1];
        eigen->eigenValues[idx*3 + 2] = eigenValues[2];
        HFnorm[idx] = sqrt(eigenValues[0]*eigenValues[0] 
                         + eigenValues[1]*eigenValues[1] 
                         + eigenValues[2]*eigenValues[2]);

        if(isnan(HFnorm[idx])){
            HFnorm[idx] = 0;
        }
        
        if(vecType == VEC_TYPE_CARTESIAN){
            for(int i = 0; i < vecSize; i++){
                eigen->eigenVectors[idx * vecSize + i] = eigenVectors[i];
            }
        }else if(vecType == VEC_TYPE_SPHERE){
            // Convert to spherical coordinates
            // zyx
            T phi, theta;
            // azimuth = atan2(y,x)
            phi = atan2(eigenVectors[1], 
                        eigenVectors[2]);
            phi = angleAdjust<T>(phi);
            // elevation = atan2(z,sqrt(x.^2 + y.^2))
            theta = atan2(eigenVectors[0], 
                sqrt(
                    eigenVectors[1] * eigenVectors[1] 
                    + eigenVectors[2] * eigenVectors[2]
                ));
            theta = angleAdjust<T>(theta);
            eigen->eigenVectors[idx * vecSize] = phi;
            eigen->eigenVectors[idx * vecSize + 1] = theta;
            eigen->eigenVectors[idx * vecSize + 2] = sqrt(
                eigenVectors[0] * eigenVectors[0] 
                + eigenVectors[1] * eigenVectors[1] 
                + eigenVectors[2] * eigenVectors[2]
            );
        }
    }
}




template __global__ void CudaHessianEigen<float>(SDM3D<float> *hessian, 
                Eigen3D<float> *eigen, float* HFnorm,
                int imgShape_0, int imgShape_1, int imgShape_2, 
                int maxIters, float tolerance, int vecType);
template __global__ void CudaHessianEigen<double>(SDM3D<double> *hessian, 
                Eigen3D<double> *eigen, double* HFnorm,
                int imgShape_0, int imgShape_1, int imgShape_2, 
                int maxIters, double tolerance, int vecType);













