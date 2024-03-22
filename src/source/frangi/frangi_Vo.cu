
#include "define.h"
#include "cuda.cuh"

__device__ float alpha_d;
__device__ float beta_d;
__device__ float c_d;
__device__ bool black_ridges_d;
extern __device__ int imgShape_d[3];

__global__ void SetFrangiParams(
    int imgShape_0, int imgShape_1, int imgShape_2,
    float alpha, float beta, float c, bool black_ridges)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;    
    if(z==0 && y==0 && x==0){
        imgShape_d[0] = imgShape_0;
        imgShape_d[1] = imgShape_1;
        imgShape_d[2] = imgShape_2;
        alpha_d = alpha;
        beta_d = beta;
        c_d = c;
        black_ridges_d = black_ridges;
    }
}

__global__ void CudaFrangiVo(Eigen3D *eigen, float* output)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;  
    if( z < imgShape_d[0] && y < imgShape_d[1] && x < imgShape_d[2] ){
        int idx = z * imgShape_d[1] * imgShape_d[2] + y * imgShape_d[2] + x;
        float e1 = eigen->eigenValues[idx*3    ];
        float e2 = eigen->eigenValues[idx*3 + 1];
        float e3 = eigen->eigenValues[idx*3 + 2];
        float Ra, Rb, S;
        
        if(abs(e3) < 1e-5) e3 = 1e-5;
        if(abs(e2) < 1e-5) e2 = 1e-5;

        Ra = abs(e2 / e3);
        Rb = abs(e1) / sqrt(abs(e2 * e3));
        S = sqrt(e1*e1 + e2*e2 + e3*e3);
        if(isnan(S)){
            S = 0;
        }

        output[idx] = (1.0 - exp( -Ra*Ra / (2* alpha_d*alpha_d)))
                    * exp( -Rb*Rb / (2* beta_d*beta_d))
                    * (1.0 - exp( -S*S / (2* c_d*c_d)));
    }
}













