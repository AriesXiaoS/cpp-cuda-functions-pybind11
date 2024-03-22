#ifndef _DEFINE_H
#define _DEFINE_H

// NAMES
namespace EIGENVEC{
    const int NONE = 0;
    const int CARTESIAN = 1;
    const int SPHERE = 2;
}

// STRUCTS
struct Conv3DParam{
    int kernel_size;
    int img_shape[3];
};
struct QRIterParam{
    int max_iters;
    float tolerance;
};

// Second Derivative Matrix 3D
struct SDM3D{
    float* xx;
    float* yy;
    float* zz;
    float* xy;
    float* xz;
    float* yz;
};

struct Eigen3D{ // 3D 图像Hessian矩阵的特征值和特征向量
    float* eigenValues;
    float* eigenVectors;
};
struct EigenTemp{
    float* A;
    float* Q;
    float* Q_temp;
    float* R;
    float* P;
    float* AK;

    float* w1;
    float* w2;
    float* H1;
    float* H2;
    float* h2;
};










#endif