
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

__device__ float normVec3(float x1, float x2, float x3){
    // return sqrt(x1*x1 + x2*x2 + x3*x3);
    return sqrt(pow(x1  , 2) + pow(x2  , 2) + pow(x3  , 2 ));
}

__device__ float normVec2(float x1, float x2){
    return sqrt(x1*x1 + x2*x2);
}

__device__ float qrSplit(float* A, float* w, float* H1, float* H2, float* h2,  float* P, float* Q, float* R){

    float x_norm, x_y_norm;

    x_norm = normVec3(A[0], A[3], A[6]);
    x_y_norm = normVec3(A[0]-x_norm, A[3], A[6]);
    w[0] = (A[0]-x_norm) / x_y_norm;
    w[1] = A[3] / x_y_norm;
    w[2] = A[6] / x_y_norm;
    // H = I - 2 * w * w.T
    H1[0] = 1 - 2 * w[0] * w[0];
    H1[1] = -2 * w[0] * w[1];
    H1[2] = -2 * w[0] * w[2];
    H1[3] = -2 * w[1] * w[0];
    H1[4] = 1 - 2 * w[1] * w[1];
    H1[5] = -2 * w[1] * w[2];
    H1[6] = -2 * w[2] * w[0];
    H1[7] = -2 * w[2] * w[1];
    H1[8] = 1 - 2 * w[2] * w[2];
    // R1 = H1 dot A
    R[0] = H1[0] * A[0] + H1[1] * A[3] + H1[2] * A[6];
    R[1] = H1[0] * A[1] + H1[1] * A[4] + H1[2] * A[7];
    R[2] = H1[0] * A[2] + H1[1] * A[5] + H1[2] * A[8];
    R[3] = H1[3] * A[0] + H1[4] * A[3] + H1[5] * A[6];
    R[4] = H1[3] * A[1] + H1[4] * A[4] + H1[5] * A[7];
    R[5] = H1[3] * A[2] + H1[4] * A[5] + H1[5] * A[8];
    R[6] = H1[6] * A[0] + H1[7] * A[3] + H1[8] * A[6];
    R[7] = H1[6] * A[1] + H1[7] * A[4] + H1[8] * A[7];
    R[8] = H1[6] * A[2] + H1[7] * A[5] + H1[8] * A[8];
    // 
    x_norm = normVec2(R[4], R[7]);
    x_y_norm = normVec2(R[4]-x_norm, R[7]);
    w[0] = (R[4]-x_norm) / x_y_norm;
    w[1] = R[7] / x_y_norm;
    // h2 = I - 2 * w * w.T
    h2[0] = 1 - 2 * w[0] * w[0];
    h2[1] = -2 * w[0] * w[1];
    h2[2] = -2 * w[1] * w[0];
    h2[3] = 1 - 2 * w[1] * w[1];
    // H2 = [E, 0; 0, h2]
    H2[0] = 1;
    H2[1] = 0;
    H2[2] = 0;
    H2[3] = 0;
    H2[4] = h2[0];
    H2[5] = h2[1];
    H2[6] = 0;
    H2[7] = h2[2];
    H2[8] = h2[3];
    // P = H2* H1
    P[0] = H2[0] * H1[0] + H2[1] * H1[3] + H2[2] * H1[6];
    P[1] = H2[0] * H1[1] + H2[1] * H1[4] + H2[2] * H1[7];
    P[2] = H2[0] * H1[2] + H2[1] * H1[5] + H2[2] * H1[8];
    P[3] = H2[3] * H1[0] + H2[4] * H1[3] + H2[5] * H1[6];
    P[4] = H2[3] * H1[1] + H2[4] * H1[4] + H2[5] * H1[7];
    P[5] = H2[3] * H1[2] + H2[4] * H1[5] + H2[5] * H1[8];
    P[6] = H2[6] * H1[0] + H2[7] * H1[3] + H2[8] * H1[6];
    P[7] = H2[6] * H1[1] + H2[7] * H1[4] + H2[8] * H1[7];
    P[8] = H2[6] * H1[2] + H2[7] * H1[5] + H2[8] * H1[8];
    // R = P dot A
    R[0] = P[0] * A[0] + P[1] * A[3] + P[2] * A[6];
    R[1] = P[0] * A[1] + P[1] * A[4] + P[2] * A[7];
    R[2] = P[0] * A[2] + P[1] * A[5] + P[2] * A[8];
    R[3] = P[3] * A[0] + P[4] * A[3] + P[5] * A[6];
    R[4] = P[3] * A[1] + P[4] * A[4] + P[5] * A[7];
    R[5] = P[3] * A[2] + P[4] * A[5] + P[5] * A[8];
    R[6] = P[6] * A[0] + P[7] * A[3] + P[8] * A[6];
    R[7] = P[6] * A[1] + P[7] * A[4] + P[8] * A[7];
    R[8] = P[6] * A[2] + P[7] * A[5] + P[8] * A[8];
    // Q = P.T
    Q[0] = P[0];
    Q[1] = P[3];
    Q[2] = P[6];
    Q[3] = P[1];
    Q[4] = P[4];
    Q[5] = P[7];
    Q[6] = P[2];
    Q[7] = P[5];
    Q[8] = P[8];
    
}
__global__ void CudaHessianEigen(SDM3D *hessian, Eigen3D *eigen, float* HFnorm)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;    
    if( z < imgShape_d[0] && y < imgShape_d[1] && x < imgShape_d[2] ){
        int idx = z * imgShape_d[1] * imgShape_d[2] + y * imgShape_d[2] + x;
        float A[9] = {hessian->xx[idx], hessian->xy[idx], hessian->xz[idx],
                      hessian->xy[idx], hessian->yy[idx], hessian->yz[idx],
                      hessian->xz[idx], hessian->yz[idx], hessian->zz[idx]};
        float eigenValues[3];
        float eigenVectors[9];
        // if(eigenVectorType_d){
            
        // }else{
        //     eigenVectors = nullptr;
        // }
        // QR 
        QREigens_3x3(A, eigenValues, eigenVectors, maxIters_d, tolerance_d);
        // printf("QREigens done \n");

        eigen->eigenValues[idx*3    ] = eigenValues[0];
        eigen->eigenValues[idx*3 + 1] = eigenValues[1];
        eigen->eigenValues[idx*3 + 2] = eigenValues[2];
        HFnorm[idx] = sqrt(eigenValues[0]*eigenValues[0] 
                         + eigenValues[1]*eigenValues[1] 
                         + eigenValues[2]*eigenValues[2]);


        /// old ///
        // float xx = hessian->xx[idx];
        // float yy = hessian->yy[idx];
        // float zz = hessian->zz[idx];
        // float xy = hessian->xy[idx];
        // float xz = hessian->xz[idx];
        // float yz = hessian->yz[idx];
        // float A[9] = {xx, xy, xz, xy, yy, yz, xz, yz, zz};
        // float Q[9] = {1,0,0,0,1,0,0,0,1};
        // float q[9], r[9], H1[9], H2[9], h2[4], P[9], w[3];
        // float e1, e2, e3, tmp;
        // //QR迭代 max_iter次
        // for(int i=0;i<30;i++){
        //     qrSplit(A, w, H1, H2, h2, P, q, r);
        //     // Q = Q dot q
        //     Q[0] = Q[0] * q[0] + Q[1] * q[3] + Q[2] * q[6];
        //     Q[1] = Q[0] * q[1] + Q[1] * q[4] + Q[2] * q[7];
        //     Q[2] = Q[0] * q[2] + Q[1] * q[5] + Q[2] * q[8];
        //     Q[3] = Q[3] * q[0] + Q[4] * q[3] + Q[5] * q[6];
        //     Q[4] = Q[3] * q[1] + Q[4] * q[4] + Q[5] * q[7];
        //     Q[5] = Q[3] * q[2] + Q[4] * q[5] + Q[5] * q[8];
        //     Q[6] = Q[6] * q[0] + Q[7] * q[3] + Q[8] * q[6];
        //     Q[7] = Q[6] * q[1] + Q[7] * q[4] + Q[8] * q[7];
        //     Q[8] = Q[6] * q[2] + Q[7] * q[5] + Q[8] * q[8];
        //     // A = r dot q
        //     A[0] = r[0] * q[0] + r[1] * q[3] + r[2] * q[6];
        //     A[1] = r[0] * q[1] + r[1] * q[4] + r[2] * q[7];
        //     A[2] = r[0] * q[2] + r[1] * q[5] + r[2] * q[8];
        //     A[3] = r[3] * q[0] + r[4] * q[3] + r[5] * q[6];
        //     A[4] = r[3] * q[1] + r[4] * q[4] + r[5] * q[7];
        //     A[5] = r[3] * q[2] + r[4] * q[5] + r[5] * q[8];
        //     A[6] = r[6] * q[0] + r[7] * q[3] + r[8] * q[6];
        //     A[7] = r[6] * q[1] + r[7] * q[4] + r[8] * q[7];
        //     A[8] = r[6] * q[2] + r[7] * q[5] + r[8] * q[8];
        // }
        // // AK = q dot r
        // A[0] = q[0] * r[0] + q[1] * r[3] + q[2] * r[6];
        // A[1] = q[0] * r[1] + q[1] * r[4] + q[2] * r[7];
        // A[2] = q[0] * r[2] + q[1] * r[5] + q[2] * r[8];
        // A[3] = q[3] * r[0] + q[4] * r[3] + q[5] * r[6];
        // A[4] = q[3] * r[1] + q[4] * r[4] + q[5] * r[7];
        // A[5] = q[3] * r[2] + q[4] * r[5] + q[5] * r[8];
        // A[6] = q[6] * r[0] + q[7] * r[3] + q[8] * r[6];
        // A[7] = q[6] * r[1] + q[7] * r[4] + q[8] * r[7];
        // A[8] = q[6] * r[2] + q[7] * r[5] + q[8] * r[8];
        // // 计算特征值
        // e1 = A[0];
        // e2 = A[4];
        // e3 = A[8];

        // if(abs(e1) > abs(e2)){
        //     tmp = e1;
        //     e1 = e2;
        //     e2 = tmp;
        // }
        // if(abs(e1) > abs(e3)){
        //     tmp = e1;
        //     e1 = e3;
        //     e3 = tmp;
        // }
        // if(abs(e2) > abs(e3)){
        //     tmp = e2;
        //     e2 = e3;
        //     e3 = tmp;
        // }
        // // // e1 e2 e3 got
        // if(e2 < 1e-5) e2 = 1e-5;
        // if(e3 < 1e-5) e3 = 1e-5;


        // eigen->eigenValues[idx*3    ] = e1;
        // eigen->eigenValues[idx*3 + 1] = e2;
        // eigen->eigenValues[idx*3 + 2] = e3;
        // HFnorm[idx] = sqrt(e1*e1 + e2*e2 + e3*e3);

        ///////////////////////////

        if(isnan(HFnorm[idx])){
            HFnorm[idx] = 0;
        }
        
        // if(eigenVectorType_d == 1){
        //     for(int i = 0; i < 9; i++){
        //         eigen->eigenVectors[idx*9 + i] = eigenVectors[i];
        //     }
        // }
        // else if(eigenVectorType_d == EIGENVEC::SPHERE){
            
        // }
    }
}


















