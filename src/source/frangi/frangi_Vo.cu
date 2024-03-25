
#include "define.h"
#include "cuda.cuh"


template <typename T>
__global__ void CudaFrangiVo(Eigen3D<T> *eigen, T* output,
    int imgShape_0, int imgShape_1, int imgShape_2, 
    T alpha, T beta, T c, bool black_ridges)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;  
    if( z < imgShape_0 && y < imgShape_1 && x < imgShape_2 ){
        int idx = z * imgShape_1 * imgShape_2 + y * imgShape_2 + x;
        T e1 = eigen->eigenValues[idx*3    ];
        T e2 = eigen->eigenValues[idx*3 + 1];
        T e3 = eigen->eigenValues[idx*3 + 2];
        T Ra, Rb, S;
        
        if(e2 < 1e-5) e2 = 1e-5;
        if(e3 < 1e-5) e3 = 1e-5; // 这里若用abs 最终会出现边缘的黑线    

        Ra = e2 / e3;
        Rb = abs(e1) / sqrt(e2 * e3);
        S = sqrt(e1*e1 + e2*e2 + e3*e3);
        if(isnan(S)){
            S = 0;
        }

        output[idx] = (1.0 - exp( -Ra*Ra / (2* alpha*alpha)))
                    * exp( -Rb*Rb / (2* beta*beta))
                    * (1.0 - exp( -S*S / (2* c*c)));
    }
}

template __global__ void CudaFrangiVo<float>(Eigen3D<float> *eigen, float* output,
    int imgShape_0, int imgShape_1, int imgShape_2, 
    float alpha, float beta, float c, bool black_ridges);
template __global__ void CudaFrangiVo<double>(Eigen3D<double> *eigen, double* output,
    int imgShape_0, int imgShape_1, int imgShape_2, 
    double alpha, double beta, double c, bool black_ridges);











