#include "define.h"
#include "cuda.cuh"

__device__ __host__ float normVec(float x1, float x2, float x3){
    return sqrt(pow(x1, 2) + pow(x2, 2) + pow(x3, 2));
}

__device__ __host__ float normVec(float x1, float x2){
    return sqrt(pow(x1, 2) + pow(x2, 2));
}

/**
*  @brief C = A dot B
*  @param A 3x3 matrix float* A[9]
*  @param B 3x3 matrix float* B[9]
*  @return C 3x3 matrix float* C[9]
*  [ 0 1 2 ]
*  [ 3 4 5 ]
*  [ 6 7 8 ]
*/
__device__ __host__ void MatrixDot3x3(float* A, float* B, float* C){
    C[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
    C[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
    C[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];
    C[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
    C[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
    C[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];
    C[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
    C[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
    C[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
}

/**
*  @brief A = A dot B
*  @param A 3x3 matrix float* A[9]
*  @param B 3x3 matrix float* B[9]
*  [ 0 1 2 ]
*  [ 3 4 5 ]
*  [ 6 7 8 ]
*/
__device__ __host__ void MatrixDotSelf3x3(float*A, float*B){
    float C[9];
    MatrixDot3x3(A, B, C);
    for(int i=0; i<9; i++){
        A[i] = C[i];
    }
}

/**
*  @brief w dot w.T -> n*n matrix
*  @param w nx1 vec float* w[n]
*  @return A n*n matrix float* A[n*n]
*/
__device__ __host__ void VecDotT(float* w, int n, float* A){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = w[i]*w[j];
        }
    }
}

/**
*  @brief A *= num
*  @param A float* A[n]
*/
__device__ __host__ void MatrixMulNum(float* A, int n, float num){
    for(int i=0; i<n; i++){
        A[i] *= num;
    }
}

/**
*  @brief A -= B
*  @param A float* A[n]
*  @param B float* B[n]
*/
__device__ __host__ void MatrixSub(float* A, float* B, int n){
    for(int i=0; i<n; i++){
        A[i] -= B[i];
    }
}

/**
*  @brief H = I - 2 * w * w.T
*  @param w nx1 vec float* w[n]
*  @param H n*n Matrix float* H[n*n]
*/
__device__ __host__ void HouseholderH(float* w, int n, float* H){
    // H <- w * w.T
    VecDotT(w, n, H);
    // H *= 2
    MatrixMulNum(H, n*n, 2);
    // H = I - H
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i==j){
                H[i*n+j] = 1 - H[i*n+j];
            }
            else{
                H[i*n+j] = -H[i*n+j];
            }
        }
    }
}

/**
*  @brief B <- A
*  @param A float* A[n]
*  @param B float* B[n]
*/
__device__ __host__ void MatrixCopy(float* A, float* B, int n){
    for(int i=0; i<n; i++){
        B[i] = A[i];
    }
}

__device__ __host__ void TransposeMatrix(float* A, int m, int n, float* B){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            B[j*m+i] = A[i*n+j];
        }
    }
}

__device__ __host__ void QRSplit_3x3(float* A, float* Q, float* R){
    //
    printf("QRSplit_3x3 before \n");
    float *w1 = new float[3]; // vec
    float *w2 = new float[2]; // vec
    float *H1 = new float[9]; // matrix 3x3
    float *H2 = new float[9]; // matrix 3x3
    float *h2 = new float[4]; // matrix 2x2
    float *P = new float[9];  // matrix 3x3
    float x_norm, x_y_norm;
    printf("QRSplit_3x3 \n");
    // 3x3
    x_norm = normVec(A[0], A[3], A[6]); // A第一列 x
    // y = [x_norm, 0, 0].T
    x_y_norm = normVec(A[0]-x_norm, A[3], A[6]);
    //计算householder反射矩阵 w = (x-y) / |x-y|
    w1[0] = (A[0]-x_norm) / x_y_norm;
    w1[1] = A[3] / x_y_norm;
    w1[2] = A[6] / x_y_norm;
    // H = E - 2 * w * w.T
    HouseholderH(w1, 3, H1);
    // Q = H1
    // R1 = H1 * A
    MatrixDot3x3(H1, A, R);
    //
    printf("2x2 \n");
    // 2x2
    x_norm = normVec(R[4], R[7]);
    x_y_norm = normVec(R[4]-x_norm, R[7]);
    w2[0] = (R[4]-x_norm) / x_y_norm;
    w2[1] = R[7] / x_y_norm;
    // h2 = E - 2 * w * w.T 
    HouseholderH(w2, 2, h2);
    // H2 = [E, 0; 0, H2]
    H2[0] = 1;  H2[1] = 0;       H2[2] = 0;
    H2[3] = 0;  H2[4] = h2[0];  H2[5] = h2[1];
    H2[6] = 0;  H2[7] = h2[2];  H2[8] = h2[3];
    // Q = H2 dot Q = H2 dot H1 (=P)
    MatrixDot3x3(H2, H1, P);
    // R2 = H2 dot R1 = H2 dot H1 dot A = P dot A
    MatrixDot3x3(P, A, R);

    // Q = P.T
    TransposeMatrix(P, 3, 3, Q);
}

__device__ __host__ void QREigens_3x3(float* A, 
                                float* eigenValues, float* eigenVectors,
                                int maxIters, float tolerance)
{
    //
    float *Q = new float[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};
    float *Q_temp = new float[9];
    float *R = new float[9];
    float *AK = new float[9];
    float temp;
    if(tolerance <= 0){
        for(int i=0; i<maxIters; i++){
            // A = QR
            QRSplit_3x3(A, Q_temp, R);
            // Q右边累乘 Q = Q * Q_temp
            MatrixDotSelf3x3(Q, Q_temp);
            // A = R * Q_temp
            MatrixDot3x3(R, Q_temp, A);
        }
        // AK = Q_temp dot R
        MatrixDot3x3(Q_temp, R, AK);
        // printf("A %f %f %f \n", A[3], A[6], A[7]);
        // printf(" no tolerance \n");
    }else{
        // printf("start iter \n");
        for(int i=0; i<maxIters; i++){
            printf("iter # %d \n", i);
            QRSplit_3x3(A, Q_temp, R);
            printf("l1 ");
            MatrixDotSelf3x3(Q, Q_temp);
            printf("l2 ");
            MatrixDot3x3(R, Q_temp, A);
            printf("l3 ");
            MatrixDot3x3(Q_temp, R, AK);
            if( abs(AK[3])<=tolerance && abs(AK[6])<=tolerance && abs(AK[7])<=tolerance){
                // printf(" iter = %d  %f %f %f \n", i, AK[3], AK[6], AK[7]);
                break;
            }
        }
        printf("iter done #\n");
    }
    printf("AK %f %f %f \n", AK[0], AK[4], AK[8]);
    eigenValues[0] = AK[0];
    eigenValues[1] = AK[4];
    eigenValues[2] = AK[8];
    printf("eigenValues %f %f %f \n", eigenValues[0], eigenValues[1], eigenValues[2]);
    // 排序 绝对值从小到大
    for(int i=0; i<2; i++){
        printf("i %d value %f \n", i, eigenValues[i]);
        for(int j=i+1; j<3; j++){
            if(abs(eigenValues[i]) > abs(eigenValues[j])){
                // 特征值
                printf(" swap ");
                temp = eigenValues[i];
                eigenValues[i] = eigenValues[j];
                eigenValues[j] = temp;
                printf(" swaped ");
                // 特征向量
                if(eigenVectors != nullptr){
                    for(int k=0; k<3; k++){
                        temp = Q[3*k+i];
                        Q[3*k+i] = Q[3*k+j];
                        Q[3*k+j] = temp;
                    }                
                }else{
                    printf("qr no vec");
                
                }
            }
        }
    }
    if(eigenVectors != nullptr){
        // eigenVectors = Q
        printf("qr vec \n");
        MatrixCopy(Q, eigenVectors, 9);    

    }else{
        printf("qr no vec");
    }
}

















