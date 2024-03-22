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

struct Hessian3D{ // 3D 图像的 Hessian 矩阵 3x3
    float* Ixx;
    float* Iyy;
    float* Izz;
    float* Ixy;
    float* Ixz;
    float* Iyz;
};
struct Eigen3D{ // 3D 图像Hessian矩阵的特征值和特征向量
    float* eigenValues;
    float* eigenVectors;
};
struct FrangiParam{
    float alpha;
    float beta;
    float gamma; // c
    bool blackRidges;
    float* sigmas; // gaussian sigmas 
    int maxIters;
    float tolerance;
    int eigenVectorType;
};










#endif