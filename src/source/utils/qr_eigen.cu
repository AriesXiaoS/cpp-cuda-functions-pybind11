#include "define.h"
#include "cuda.cuh"

template <typename T>
__device__ __host__ T normVec(T x1, T x2, T x3){
    return sqrt(pow(x1, 2) + pow(x2, 2) + pow(x3, 2));
}

template <typename T>
__device__ __host__ T normVec(T x1, T x2){
    return sqrt(pow(x1, 2) + pow(x2, 2));
}

/**
*  @brief C = A dot B
*  @param A 3x3 matrix T* A[9]
*  @param B 3x3 matrix T* B[9]
*  @return C 3x3 matrix T* C[9]
*  [ 0 1 2 ]
*  [ 3 4 5 ]
*  [ 6 7 8 ]
*/
template <typename T>
__device__ __host__ void MatrixDot3x3(T* A, T* B, T* C){
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
*  @param A 3x3 matrix T* A[9]
*  @param B 3x3 matrix T* B[9]
*  [ 0 1 2 ]
*  [ 3 4 5 ]
*  [ 6 7 8 ]
*/
template <typename T>
__device__ __host__ void MatrixDotSelf3x3(T*A, T*B){
    T C[9];
    MatrixDot3x3(A, B, C);
    for(int i=0; i<9; i++){
        A[i] = C[i];
    }
}

/**
*  @brief w dot w.T -> n*n matrix
*  @param w nx1 vec T* w[n]
*  @return A n*n matrix T* A[n*n]
*/
template <typename T>
__device__ __host__ void VecDotT(T* w, int n, T* A){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = w[i]*w[j];
        }
    }
}

/**
*  @brief A *= num
*  @param A T* A[n]
*/
template <typename T>
__device__ __host__ void MatrixMulNum(T* A, int n, T num){
    for(int i=0; i<n; i++){
        A[i] *= num;
    }
}

/**
*  @brief A -= B
*  @param A T* A[n]
*  @param B T* B[n]
*/
template <typename T>
__device__ __host__ void MatrixSub(T* A, T* B, int n){
    for(int i=0; i<n; i++){
        A[i] -= B[i];
    }
}

/**
*  @brief H = I - 2 * w * w.T
*  @param w nx1 vec T* w[n]
*  @param H n*n Matrix T* H[n*n]
*/
template <typename T>
__device__ __host__ void HouseholderH(T* w, int n, T* H){
    // H <- w * w.T
    VecDotT<T>(w, n, H);
    // H *= 2
    MatrixMulNum<T>(H, n*n, 2);
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
*  @param A T* A[n]
*  @param B T* B[n]
*/
template <typename T>
__device__ __host__ void MatrixCopy(T* A, T* B, int n){
    for(int i=0; i<n; i++){
        B[i] = A[i];
    }
}

template <typename T>
__device__ __host__ void TransposeMatrix(T* A, int m, int n, T* B){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            B[j*m+i] = A[i*n+j];
        }
    }
}

template <typename T>
__device__ __host__ void QRSplit_3x3(T* A, T* Q, T* R){
    //
    T w1[3],w2[2],H1[9],H2[9],h2[4],P[9];

    // T *w1 = new T[3]; // vec
    // T *w2 = new T[2]; // vec
    // T *H1 = new T[9]; // matrix 3x3
    // T *H2 = new T[9]; // matrix 3x3
    // T *h2 = new T[4]; // matrix 2x2
    // T *P = new T[9];  // matrix 3x3
    T x_norm, x_y_norm;
    // 3x3
    x_norm = normVec<T>(A[0], A[3], A[6]); // A第一列 x
    // y = [x_norm, 0, 0].T
    x_y_norm = normVec<T>(A[0]-x_norm, A[3], A[6]);
    //计算householder反射矩阵 w = (x-y) / |x-y|
    w1[0] = (A[0]-x_norm) / x_y_norm;
    w1[1] = A[3] / x_y_norm;
    w1[2] = A[6] / x_y_norm;
    // H = E - 2 * w * w.T
    HouseholderH<T>(w1, 3, H1);
    // Q = H1
    // R1 = H1 * A
    MatrixDot3x3<T>(H1, A, R);
    //
    // 2x2
    x_norm = normVec<T>(R[4], R[7]);
    x_y_norm = normVec<T>(R[4]-x_norm, R[7]);
    w2[0] = (R[4]-x_norm) / x_y_norm;
    w2[1] = R[7] / x_y_norm;
    // h2 = E - 2 * w * w.T 
    HouseholderH<T>(w2, 2, h2);
    // H2 = [E, 0; 0, H2]
    H2[0] = 1;  H2[1] = 0;       H2[2] = 0;
    H2[3] = 0;  H2[4] = h2[0];  H2[5] = h2[1];
    H2[6] = 0;  H2[7] = h2[2];  H2[8] = h2[3];
    // Q = H2 dot Q = H2 dot H1 (=P)
    MatrixDot3x3<T>(H2, H1, P);
    // R2 = H2 dot R1 = H2 dot H1 dot A = P dot A
    MatrixDot3x3<T>(P, A, R);

    // Q = P.T
    TransposeMatrix<T>(P, 3, 3, Q);
}

template <typename T>
__device__ __host__ void QREigens_3x3(T* A, 
                                T* eigenValues, T* eigenVectors,
                                int maxIters, T tolerance, int vecSize)
{
    //
    T Q[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    T Q_temp[9];
    T R[9];
    T AK[9];
    T temp;
    if(tolerance <= 0){
        for(int i=0; i<maxIters; i++){
            // A = QR
            QRSplit_3x3<T>(A, Q_temp, R);
            // Q右边累乘 Q = Q * Q_temp
            MatrixDotSelf3x3<T>(Q, Q_temp);
            // A = R * Q_temp
            MatrixDot3x3<T>(R, Q_temp, A);
        }
        // AK = Q_temp dot R
        MatrixDot3x3<T>(Q_temp, R, AK);
    }else{
        for(int i=0; i<maxIters; i++){
            QRSplit_3x3<T>(A, Q_temp, R);
            MatrixDotSelf3x3<T>(Q, Q_temp);
            MatrixDot3x3<T>(R, Q_temp, A);
            MatrixDot3x3<T>(Q_temp, R, AK);
            if( abs(AK[3])<=tolerance && abs(AK[6])<=tolerance && abs(AK[7])<=tolerance){
                break;
            }
        }
    }
    eigenValues[0] = AK[0];
    eigenValues[1] = AK[4];
    eigenValues[2] = AK[8];
    // 排序 绝对值从小到大
    for(int i=0; i<2; i++){
        for(int j=i+1; j<3; j++){
            if(abs(eigenValues[i]) > abs(eigenValues[j])){
                // 特征值
                temp = eigenValues[i];
                eigenValues[i] = eigenValues[j];
                eigenValues[j] = temp;
                // 特征向量
                if(vecSize > 0){
                    for(int k=0; k<3; k++){
                        temp = Q[3*k+i];
                        Q[3*k+i] = Q[3*k+j];
                        Q[3*k+j] = temp;
                    }                
                }
            }
        }
    }
    if(vecSize > 0){
        // eigenVectors <= Q 这里都是 VEC_TYPE_CARTESIAN
        MatrixCopy(Q, eigenVectors, 3);    
    }
}


template __device__ __host__ void QRSplit_3x3<float>(float* A, float* Q, float* R);
template __device__ __host__ void QRSplit_3x3<double>(double* A, double* Q, double* R);

template __device__ __host__ void QREigens_3x3<float>(float* A, 
float* eigenValues, float* eigenVectors, int maxIters, float tolerance, int vecSize);
template __device__ __host__ void QREigens_3x3<double>(double* A, 
double* eigenValues, double* eigenVectors, int maxIters, double tolerance, int vecSize);















