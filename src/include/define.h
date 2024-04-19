#ifndef _DEFINE_H
#define _DEFINE_H


#define VEC_TYPE_NONE 0
#define VEC_TYPE_CARTESIAN 1
#define VEC_TYPE_SPHERE 2

#define VEC_TYPE_CARTESIAN_SIZE 3
#define VEC_TYPE_SPHERE_SIZE 3

#define PI 3.1415926535


// STRUCTS
struct Conv3DParam{
    int kernel_size;
    int img_shape[3];
};


// Second Derivative Matrix 3D
template <typename T>
struct SDM3D{
    T* xx;
    T* yy;
    T* zz;
    T* xy;
    T* xz;
    T* yz;
};

template <typename T>
struct Eigen3D{ // 3D 图像Hessian矩阵的特征值和特征向量
    T* eigenValues;
    T* eigenVectors;
};











#endif